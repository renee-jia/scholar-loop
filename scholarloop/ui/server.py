"""Loopback-only HTTP service with same-origin, token-protected mutations."""
from __future__ import annotations

import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scholarloop.ui.catalog import catalog
from scholarloop.ui.jobs import Jobs
from scholarloop.ui.store import Store

STATIC = Path(__file__).parent / "static"


class Application:
    def __init__(self, workspace, runs_dir):
        self.store = Store(workspace, runs_dir)
        self.jobs = Jobs(self.store)
        self.token = secrets.token_urlsafe(32)

    def bootstrap(self):
        return {"runs": self.store.list_runs(), "profiles": self.store.profiles(), "actions": catalog(),
                "environment": self.store.environment(), "connected": bool(self.jobs.api_key)}


def make_server(app, host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("This local research UI can only bind to 127.0.0.1 or localhost")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Artifact paths and user input need not be repeated in the terminal.
            pass

        def send(self, status, body, content_type="application/json; charset=utf-8", filename=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def check_origin(self, mutation=False):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                raise PermissionError("Invalid host")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in hosts}:
                raise PermissionError("Only same-origin requests are allowed")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise PermissionError("Cross-site requests are not allowed")
            if mutation and not secrets.compare_digest(self.headers.get("X-ScholarLoop-Token", ""), app.token):
                raise PermissionError("Reload this page before trying again")

        def do_GET(self):
            try:
                self.check_origin()
                url = urlsplit(self.path)
                parts = url.path.strip("/").split("/")
                query = parse_qs(url.query)
                if url.path == "/":
                    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.token)
                    return self.send(200, html, "text/html; charset=utf-8")
                if url.path in {"/app.js", "/style.css", "/favicon.svg"}:
                    p = STATIC / url.path[1:]
                    return self.send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] + "; charset=utf-8")
                if url.path == "/api/bootstrap":
                    return self.send(200, app.bootstrap())
                if url.path == "/api/runs":
                    return self.send(200, app.store.list_runs())
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    ident = parts[2]
                    if len(parts) == 3:
                        detail = app.store.detail(ident)
                        detail["jobs"] = app.jobs.list(ident)
                        return self.send(200, detail)
                    if len(parts) == 4 and parts[3] == "artifact":
                        return self.send(200, {"text": app.store.artifact(ident, query.get("path", [""])[0])})
                    if len(parts) == 4 and parts[3] == "export":
                        return self.send(200, app.store.export(ident), "application/zip", filename="scholarloop-evidence.zip")
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "log":
                    return self.send(200, app.jobs.log(parts[2], int(query.get("offset", ["0"])[0])))
                self.send(404, {"error": "Not found"})
            except PermissionError as exc:
                self.send(403, {"error": str(exc)})
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self):
            try:
                self.check_origin(mutation=True)
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                parts = urlsplit(self.path).path.strip("/").split("/")
                if parts == ["api", "runs"]:
                    return self.send(201, app.store.create(data))
                if parts == ["api", "import"]:
                    return self.send(201, app.store.import_run(data.get("path", ""), data.get("profile")))
                if parts == ["api", "sample"]:
                    return self.send(201, app.store.import_run(app.store.workspace / "examples/sample_runs/digits", readonly=True))
                if parts == ["api", "connection"]:
                    key = data.get("api_key", "")
                    if not isinstance(key, str) or len(key) > 1024:
                        raise ValueError("Invalid API key")
                    app.jobs.api_key = key.strip()
                    return self.send(200, {"connected": bool(app.jobs.api_key)})
                if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                    if parts[3] == "delete":
                        if data.get("confirmed") is not True:
                            raise ValueError("Confirm deletion before removing this research")
                        # Serialize with job creation so a task cannot start during deletion.
                        with app.jobs.lock:
                            if app.jobs.active(parts[2]):
                                raise ValueError("Stop the running task before deleting this research")
                            result = app.store.delete(parts[2])
                        return self.send(200, result)
                    if parts[3] == "actions":
                        return self.send(202, app.jobs.start(parts[2], data.get("action"), data.get("params", {})))
                    if parts[3] == "settings":
                        if app.jobs.active(parts[2]):
                            raise ValueError("Wait for the current task to finish before changing settings")
                        return self.send(200, app.store.update(parts[2], data))
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                    return self.send(200, app.jobs.cancel(parts[2]))
                self.send(404, {"error": "Not found"})
            except PermissionError as exc:
                self.send(403, {"error": str(exc)})
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)
