"""Start with: python -m scholarloop.ui"""
import argparse
from contextlib import ExitStack
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


def run_utf8(argv):
    """Use UTF-8 consistently and keep the launcher attached to its terminal."""
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "scholarloop.ui", *argv])
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Both processes share the console and receive Ctrl+C. Let the UI
            # finish cancelling its workers and releasing its directory lock.
            continue


def main(argv=None):
    parser = argparse.ArgumentParser(description="ScholarLoop local research workspace")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="checkout containing profiles/ and engines/")
    parser.add_argument("--runs-dir", type=Path, help="research directory (default: <workspace>/runs)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    workspace = args.workspace.resolve()
    if not (workspace / "profiles").is_dir() or not (workspace / "engines").is_dir():
        parser.error("run from the ScholarLoop checkout, or pass --workspace PATH")
    if not sys.flags.utf8_mode:
        # Existing research modules use the interpreter's default text encoding.
        # Do not depend on the system locale, including on macOS and Linux.
        return run_utf8(sys.argv[1:] if argv is None else argv)
    # All worker subprocesses use UTF-8; the application also writes multilingual artifacts.
    os.environ["PYTHONUTF8"] = "1"
    try:
        from scholarloop.ui.server import Application, make_server
        from scholarloop.core.locking import exclusive_lock
        runs = (args.runs_dir or workspace / "runs").resolve()
        runs.mkdir(parents=True, exist_ok=True)
        with ExitStack() as cleanup:
            try:
                cleanup.enter_context(exclusive_lock(runs / ".ui-server.lock"))
            except BlockingIOError:
                print(f"The local UI is already running for this research directory:\n{runs}\n"
                      "Use the existing UI, or press Ctrl+C in its original terminal before restarting.\n"
                      "Changing the port will not release this directory lock.", file=sys.stderr)
                return 1
            app = Application(workspace, runs)
            cleanup.callback(app.jobs.close)
            try:
                server = make_server(app, args.host, args.port)
            except OSError as exc:
                print(f"Cannot listen on {args.host}:{args.port}: {exc}.\n"
                      "The port may be in use or reserved by the operating system. Stop the other service, "
                      "or use --port 0 to choose an available port.", file=sys.stderr)
                return 1
            cleanup.callback(server.server_close)
            url = f"http://{args.host}:{server.server_address[1]}"
            print(f"ScholarLoop workspace: {url}\nResearch directory: {runs}\nPress Ctrl+C to stop.", flush=True)
            if not args.no_browser:
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
            try:
                server.serve_forever(poll_interval=0.3)
            except KeyboardInterrupt:
                pass
        return 0
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Install with: pip install -e ".[dev,llm]"', file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Cannot access the local UI files: {exc}. Check write access to the research directory.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
