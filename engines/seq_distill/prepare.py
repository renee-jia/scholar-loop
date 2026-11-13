"""FROZEN module for the seq-distill engine: task, teacher, student architecture and evaluation.
Never edit this file. It is the trust anchor:

  1. `load_train()` hands the agent ONLY the training prompts; val and test prompts stay here.
  2. `load_teacher()` returns the frozen teacher. train.py may query it on any token sequence,
     including sequences the student itself generated, but cannot change it.
  3. `build_model()` is frozen and the student's size is fixed here (STUDENT), so no run can
     change capacity: this engine compares training procedures, not model sizes.
  4. `score(artifact)` loads a state_dict with `weights_only=True`, decodes greedily on a frozen
     split and emits the trusted number from pristine ROOT.

Task: a multi-modal string transduction. The input is L=6 symbols from an alphabet of 8. Three
outputs are equally valid: the input copied, the input reversed, and the input reversed with
every symbol incremented (mod 8). The teacher is a 2-layer transformer trained on inputs paired
with one valid output picked uniformly at random, so it models a three-mode distribution. A
student is scored by decoding greedily from the prompt and checking whether the output is one
of the three valid strings; a mixture of modes is invalid. The metric is the invalid rate (%).

A sequence is `x_1..x_L SEP y_1..y_L`. Symbols are token ids 0..7 and SEP is 8. The teacher is
trained once, deterministically, and cached under ~/.cache/scholarloop.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

METRIC_NAME = "val_invalid_rate"
N_FOLDS = 5
ALPHABET, LENGTH = 8, 6
SEP, VOCAB = ALPHABET, ALPHABET + 1
SEQ_LEN = 2 * LENGTH + 1
N_MODES = 3
_FOLD_SIZE, _TEST = 500, 2000
_TEACHER = {"d_model": 96, "n_layers": 2}
STUDENT = {"d_model": 32, "n_layers": 2}     # fixed: small enough that training procedures differ
# The step budget every condition is expected to use ($SCHOLARLOOP_DISTILL_STEPS shortens it in tests).
TRAIN_STEPS = int(os.environ.get("SCHOLARLOOP_DISTILL_STEPS", "3000"))
_TEACHER_STEPS, _TEACHER_BATCH, _TEACHER_LR = 8000, 128, 1e-3
_cache: dict = {}


def _teacher_path() -> Path:
    root = os.environ.get("SCHOLARLOOP_DATA_DIR") or Path.home() / ".cache" / "scholarloop"
    return Path(root) / "seq-distill-teacher-v1.pt"


def _decode_index(index: np.ndarray) -> np.ndarray:
    """Input number `index` in [0, ALPHABET**LENGTH) as its LENGTH base-ALPHABET digits."""
    return np.stack([(index // ALPHABET ** k) % ALPHABET for k in range(LENGTH)], 1).astype("int64")


def _inputs() -> dict:
    """Every possible input, shuffled once: the first N_FOLDS * _FOLD_SIZE form the pool the folds
    are cut from, the next _TEST are the held-out test, and the teacher trains on the rest."""
    if "inputs" not in _cache:
        order = np.random.RandomState(0).permutation(ALPHABET ** LENGTH)
        pool = N_FOLDS * _FOLD_SIZE
        _cache["inputs"] = {"pool": _decode_index(order[:pool]), "test": _decode_index(order[pool:pool + _TEST]),
                            "teacher": _decode_index(order[pool + _TEST:])}
    return _cache["inputs"]


def _split(fold: int = 0):
    """Training prompts, val prompts (fold `fold` of the pool) and the held-out test prompts."""
    d = _inputs()
    folds = np.arange(len(d["pool"])) // _FOLD_SIZE
    return d["pool"][folds != fold], d["pool"][folds == fold], d["test"]


def _fold_from_env() -> int:
    return int(os.environ.get("SCHOLARLOOP_FOLD", "0"))


def load_train():
    """Training prompts only: int64 symbols of shape [N, LENGTH]. No outputs are given; ask the
    teacher, or use `valid_outputs` for hard labels."""
    return _split(_fold_from_env())[0]


def valid_outputs(x):
    """The N_MODES valid outputs of each input, shape [N_MODES, N, LENGTH] (torch tensor in and out)."""
    import torch
    return torch.stack([x, x.flip(1), (x.flip(1) + 1) % ALPHABET])


def prompt(x):
    """`x` [N, LENGTH] followed by SEP: the prefix a model continues."""
    import torch
    return torch.cat([x, torch.full((len(x), 1), SEP, dtype=x.dtype)], 1)


def build_model(hparams: dict | None = None):
    """The frozen student: a pre-norm causal transformer of the fixed size STUDENT (2 heads).
    `hparams` is accepted for the engine interface and ignored. `model(ids)` returns next-token
    logits of shape [N, len, VOCAB]."""
    return _transformer(STUDENT)


def _transformer(size: dict):
    import torch
    import torch.nn as nn

    class CausalTransformer(nn.Module):
        def __init__(self, d: int, layers: int):
            super().__init__()
            self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(SEQ_LEN, d)
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                d, 2, 4 * d, dropout=0.0, batch_first=True, norm_first=True) for _ in range(layers)])
            self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, VOCAB)

        def forward(self, ids):
            n = ids.shape[1]
            h = self.tok(ids) + self.pos(torch.arange(n))
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
            for block in self.blocks:
                h = block(h, src_mask=mask)
            return self.out(self.norm(h))

    torch.manual_seed(0)    # construction is deterministic; train.py reseeds before initialising
    return CausalTransformer(int(size["d_model"]), int(size["n_layers"]))


def generate(model, x, *, sample: bool = False, temperature: float = 1.0, generator=None):
    """Continue each prompt for LENGTH output symbols and return the full sequences [N, SEQ_LEN].
    Greedy by default; with `sample`, draws from the model at `temperature`. No gradients."""
    import torch

    ids = prompt(x)
    with torch.no_grad():
        for _ in range(LENGTH):
            logits = model(ids)[:, -1, :ALPHABET]
            if sample:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
    return ids


def _train_teacher(path: Path) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    inputs = torch.from_numpy(_inputs()["teacher"])          # disjoint from the pool and the test
    model = _transformer(_TEACHER)
    opt = torch.optim.AdamW(model.parameters(), _TEACHER_LR)
    for _ in range(_TEACHER_STEPS):
        x = inputs[torch.randint(0, len(inputs), (_TEACHER_BATCH,), generator=g)]
        mode = torch.randint(0, N_MODES, (_TEACHER_BATCH,), generator=g)
        y = valid_outputs(x)[mode, torch.arange(_TEACHER_BATCH)]
        ids = torch.cat([prompt(x), y], 1)
        logits = model(ids[:, :-1])[:, LENGTH:, :ALPHABET]
        loss = F.cross_entropy(logits.reshape(-1, ALPHABET), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model.state_dict()), tmp)
    os.replace(tmp, path)


def load_teacher():
    """The frozen teacher, in eval mode with gradients off. Trained and cached on first use."""
    import torch

    if "teacher" not in _cache:
        path = _teacher_path()
        if not path.exists():
            _train_teacher(path)
        model = _transformer(_TEACHER)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _cache["teacher"] = model
    return _cache["teacher"]


def evaluate(model, split: str = "val", fold: int = 0) -> float:
    """Invalid rate (%) of greedy decodes on a frozen split: the share of prompts whose output is
    none of the valid strings. Lower is better."""
    import torch

    _x_tr, x_val, x_te = _split(fold)
    x = torch.from_numpy(x_te if split == "test" else x_val)
    model.eval()
    y = generate(model, x)[:, LENGTH + 1:]
    valid = (valid_outputs(x) == y).all(-1).any(0)
    return round(float((~valid).float().mean() * 100.0), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None,
                diagnostics: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}, "diagnostics": diagnostics or {}}))


def score(artifact_path: str, split: str = "val", fold: int = 0) -> None:
    """Trusted metric: rebuild the frozen student from config, load the agent's weights (safely),
    and score greedy decodes on a frozen split."""
    import torch

    bundle = torch.load(artifact_path, weights_only=True)   # state_dict only, no code execution
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    # Numbers train.py chose to report about its own run (e.g. the step a switch fired). They are
    # passed through as self-reported diagnostics and play no part in the metric.
    reported = bundle.get("diagnostics") or {}
    diagnostics = {str(k): float(v) for k, v in reported.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")}
    emit_result(evaluate(model, split, fold), config=cfg, diagnostics=diagnostics)


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
        print("usage: python -m engines.seq_distill.prepare score <artifact> [test | val [fold]]",
              file=sys.stderr)
        return 2
    score(*parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
