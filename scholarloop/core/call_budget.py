"""A persistent request cap, checked before every model attempt (including retries).

This limits API requests, not dollars/tokens. Reserve before dispatch so a crash or
transport failure cannot silently restore budget. Raising the cap preserves usage.
"""

from pathlib import Path
from threading import Lock

from scholarloop.core.llm import LLMClient
from scholarloop.core.locking import exclusive_lock
from scholarloop.review.contracts import atomic_json, read_json


class CallBudgetExceeded(RuntimeError):
    pass


class CallLimitedLLM(LLMClient):
    def __init__(self, client, max_calls, *, path=None):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.client, self.max_calls = client, max_calls
        self.path = Path(path) if path else None
        self._used, self._mutex = 0, Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        return getattr(self.client, "model", None)

    @property
    def usage(self):
        return getattr(self.client, "usage", {"calls": self._used})

    def _reserve(self):
        with self._mutex:
            if self.path:
                with exclusive_lock(self.path.with_suffix(".lock")):
                    state = read_json(self.path) if self.path.exists() else {"reserved_calls": 0}
                    used = state["reserved_calls"]
                    if type(used) is not int or used < 0:
                        raise CallBudgetExceeded("Invalid persisted request budget; inspect the accounting record")
                    if used >= self.max_calls:
                        raise CallBudgetExceeded(f"LLM request budget exhausted ({used}/{self.max_calls}); "
                                                 "saved work is retained; raise the cap explicitly to continue")
                    atomic_json(self.path, {"reserved_calls": used + 1, "max_calls": self.max_calls})
            elif self._used >= self.max_calls:
                raise CallBudgetExceeded(f"LLM request budget exhausted ({self._used}/{self.max_calls})")
            self._used += 1

    def complete(self, prompt, *, system=None, max_tokens=16000):
        self._reserve()
        return self.client.complete(prompt, system=system, max_tokens=max_tokens)

    def complete_json(self, prompt, schema, *, system=None, max_tokens=16000, context=None):
        self._reserve()
        extra = {} if context is None else {"context": context}
        return self.client.complete_json(prompt, schema, system=system, max_tokens=max_tokens, **extra)
