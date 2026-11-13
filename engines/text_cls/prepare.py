"""FROZEN module for the agnews-bow engine: data, model construction, and evaluation.
Never edit this file. It is the trust anchor:

  1. `load_train()` hands the agent ONLY the training folds; val and test never leave here.
  2. `build_model()` is frozen, so the model the scorer reconstructs is defined by trusted code.
     The agent shapes it through hyperparameters and changes the training procedure in train.py.
  3. `score(artifact)` loads a state_dict with `weights_only=True` into the frozen architecture,
     evaluates on a frozen split, and emits the trusted number from pristine ROOT.

Data: a fixed, class-stratified subset of AG News (Zhang et al., 2015): 8,000 training-split
articles as the cross-validation pool and 2,000 official-test articles as the held-out test.
Text is lower-cased, split on non-alphanumerics, truncated to 64 tokens and hashed into 20,000
buckets (id 0 is padding). The subset is built once from the Hugging Face copy and cached.
"""

from __future__ import annotations

import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np

METRIC_NAME = "val_top1_err"
N_FOLDS = 5
N_CLASSES = 4
VOCAB = 20000          # hash buckets; token ids are 1..VOCAB, 0 pads
MAX_LEN = 64
_POOL, _TEST = 8000, 2000
_DATASET = "fancyzhx/ag_news"
_TOKEN = re.compile(r"[a-z0-9]+")
_cache: dict = {}


def encode(text: str) -> np.ndarray:
    """Hashed token ids for one text, padded to MAX_LEN. Stable across processes (crc32)."""
    ids = [zlib.crc32(t.encode()) % VOCAB + 1 for t in _TOKEN.findall(text.lower())[:MAX_LEN]]
    return np.array(ids + [0] * (MAX_LEN - len(ids)), dtype="int64")


def _data_path() -> Path:
    root = os.environ.get("SCHOLARLOOP_DATA_DIR") or Path.home() / ".cache" / "scholarloop"
    return Path(root) / "agnews-bow-v1.npz"


def _stratified(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    per = n // N_CLASSES
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), per, replace=False)
                                   for c in range(N_CLASSES)]))


def _build(path: Path) -> None:
    from datasets import load_dataset   # only needed the first time; the subset is cached after

    raw = load_dataset(_DATASET)
    out = {}
    for name, split, n in (("pool", "train", _POOL), ("test", "test", _TEST)):
        labels = np.array(raw[split]["label"], dtype="int64")
        keep = _stratified(labels, n, seed=0)
        texts = raw[split].select(keep.tolist())["text"]
        out[f"x_{name}"] = np.stack([encode(t) for t in texts])
        out[f"y_{name}"] = labels[keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def _arrays() -> dict:
    if not _cache:
        path = _data_path()
        if not path.exists():
            _build(path)
        with np.load(path) as data:
            _cache.update({k: data[k] for k in data.files})
    return _cache


def _split(fold: int = 0):
    """Train / val / held-out test. The pool is cut into N_FOLDS stratified folds: `fold` is the
    val split the loop selects on, the rest is training data. The test split is scored only on
    full-tier runs and never selected on."""
    from sklearn.model_selection import StratifiedKFold

    d = _arrays()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(d["x_pool"], d["y_pool"]))
    tr, va = folds[fold]
    return d["x_pool"][tr], d["x_pool"][va], d["x_test"], d["y_pool"][tr], d["y_pool"][va], d["y_test"]


def _fold_from_env() -> int:
    return int(os.environ.get("SCHOLARLOOP_FOLD", "0"))


def load_train():
    """Training data only: (token ids [N, MAX_LEN] with 0 as padding, labels [N])."""
    x_tr, _x_val, _x_te, y_tr, _y_val, _y_te = _split(_fold_from_env())
    return x_tr, y_tr


def build_model(hparams: dict):
    """Frozen architecture: mean of token embeddings (padding masked) → `depth` hidden layers of
    width `hidden` → class logits. `embed_dim`, `hidden` and `depth` are hyperparameters."""
    import torch
    import torch.nn as nn

    class BagOfEmbeddings(nn.Module):
        def __init__(self, dim: int, hidden: int, depth: int):
            super().__init__()
            self.embed = nn.Embedding(VOCAB + 1, dim, padding_idx=0)
            layers, d = [], dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            self.head = nn.Sequential(*layers, nn.Linear(d, N_CLASSES))

        def pool(self, ids):
            mask = (ids != 0).unsqueeze(-1).to(self.embed.weight.dtype)
            return (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        def forward(self, ids):
            return self.head(self.pool(ids))

    torch.manual_seed(0)    # construction is deterministic; train.py reseeds before initialising
    return BagOfEmbeddings(int(hparams["embed_dim"]), int(hparams["hidden"]), int(hparams["depth"]))


def evaluate(model, split: str = "val", fold: int = 0) -> float:
    """Top-1 error (%) on a frozen split (val fold `fold`, or the held-out test). Lower is better."""
    import torch

    _x_tr, x_val, x_te, _y_tr, y_val, y_te = _split(fold)
    x, y = (x_te, y_te) if split == "test" else (x_val, y_val)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).argmax(1).numpy()
    return round(float((pred != y).mean() * 100.0), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str, split: str = "val", fold: int = 0) -> None:
    """Trusted metric: rebuild the frozen model from config, load the agent's weights (safely),
    and score on a frozen split."""
    import torch

    bundle = torch.load(artifact_path, weights_only=True)   # state_dict only, no code execution
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    emit_result(evaluate(model, split, fold), config=cfg)


def parse_score_args(argv: list[str]):
    """`score <artifact> [val|test] [fold]` -> (artifact, split, fold), or None if malformed."""
    if not 2 <= len(argv) <= 4 or argv[0] != "score":
        return None
    split = argv[2] if len(argv) > 2 else "val"
    fold = argv[3] if len(argv) > 3 else "0"
    if split not in ("val", "test") or not fold.isdigit() or int(fold) >= N_FOLDS:
        return None
    return argv[1], split, int(fold)


def _main(argv: list[str]) -> int:
    parsed = parse_score_args(argv)
    if parsed is None:
        print("usage: python -m engines.text_cls.prepare score <artifact> [test | val [fold]]",
              file=sys.stderr)
        return 2
    score(*parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
