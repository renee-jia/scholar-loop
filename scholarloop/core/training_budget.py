"""Durable training-attempt reservations; failures and crashes never refund compute."""
from pathlib import Path
from threading import Lock

from scholarloop.core.locking import exclusive_lock
from scholarloop.review.contracts import atomic_json, read_json


class TrainingBudgetExceeded(RuntimeError):
    pass


class TrainingBudget:
    def __init__(self, path, limit, *, initial_used=0):
        if type(limit) is not int or limit < 0 or type(initial_used) is not int or initial_used < 0:
            raise ValueError("Training budget and initial usage must be nonnegative integers")
        self.path, self.limit, self._mutex = Path(path), limit, Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(self.path.with_suffix(".lock")):
            if not self.path.exists():
                atomic_json(self.path, {"version": 1, "reserved": initial_used})
            elif self.used < initial_used:
                raise TrainingBudgetExceeded("Training budget disagrees with recorded execution")

    @property
    def used(self):
        state = read_json(self.path)
        if state.get("version") != 1 or type(state.get("reserved")) is not int or state["reserved"] < 0:
            raise TrainingBudgetExceeded("Invalid training budget accounting")
        return state["reserved"]

    def reserve(self):
        with self._mutex, exclusive_lock(self.path.with_suffix(".lock")):
            used = self.used
            if used >= self.limit:
                raise TrainingBudgetExceeded(f"Training budget exhausted ({used}/{self.limit})")
            atomic_json(self.path, {"version": 1, "reserved": used + 1})
