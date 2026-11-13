"""EDITABLE training script for the seq-distill engine.

Builds the (frozen) student via `prepare.build_model`, distills the frozen teacher into it on the
FROZEN training prompts (`prepare.load_train()` never returns val or test), and saves a
`state_dict` + config to $SCHOLARLOOP_ARTIFACT. It does NOT compute or report the metric: the
runner runs the frozen `prepare score` step from pristine ROOT on the artifact. The runner
injects a hyperparameter override via $SCHOLARLOOP_CONFIG.

The default is word-level distillation. Each step draws a batch of prompts, obtains output
sequences for them, and trains the student to match the teacher's next-token distribution at
every output position (forward KL). `on_policy_frac` decides where the output sequences come
from: with that probability a batch is sampled from the STUDENT (on-policy: the teacher labels
the prefixes the student actually produces), otherwise from the TEACHER (off-policy). 0.0 is
pure off-policy, 1.0 pure on-policy.

Everything about the training procedure is yours to change here: who generates the sequences
and how (sampling temperature, greedy, mixing schedules, replay), the divergence, which
positions or sequences are weighted, the optimizer and schedule. Useful frozen helpers:
`prepare.load_teacher()`, `prepare.generate(model, x, sample=..., temperature=..., generator=...)`,
`prepare.prompt(x)`, `prepare.valid_outputs(x)` (hard labels for the training prompts),
and the constants LENGTH, ALPHABET, SEP. Sequences are `x_1..x_L SEP y_1..y_L`.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from engines.seq_distill import prepare

HPARAMS = {
    "lr": 0.003,
    "d_model": 32,
    "n_layers": 2,
    "steps": 3000,
    "on_policy_frac": 0.0,
    "temperature": 1.0,
}
BATCH = 64
L, A = prepare.LENGTH, prepare.ALPHABET


def effective_hparams() -> dict:
    return {**HPARAMS, **json.loads(os.environ.get("SCHOLARLOOP_CONFIG", "{}"))}


def output_logits(model, ids):
    """Next-token logits over symbols at the L output positions of full sequences `ids`."""
    return model(ids[:, :-1])[:, L:, :A]


def train_model(seed: int, h: dict):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    prompts = torch.from_numpy(prepare.load_train())     # training prompts only
    teacher = prepare.load_teacher()                     # frozen
    student = prepare.build_model(h)                     # frozen architecture, shaped by hparams
    torch.manual_seed(seed)
    for module in student.modules():                     # seed-dependent initialisation
        reset = getattr(module, "reset_parameters", None) or getattr(module, "_reset_parameters", None)
        if reset is not None:
            reset()
    opt = torch.optim.AdamW(student.parameters(), lr=float(h["lr"]))
    tau = float(h["temperature"])
    for _ in range(int(h["steps"])):
        x = prompts[torch.randint(0, len(prompts), (BATCH,), generator=g)]
        on_policy = torch.rand(1, generator=g).item() < float(h["on_policy_frac"])
        student.eval()
        ids = prepare.generate(student if on_policy else teacher, x, sample=True, generator=g)
        student.train()
        with torch.no_grad():
            target = F.softmax(output_logits(teacher, ids) / tau, -1)
        log_q = F.log_softmax(output_logits(student, ids) / tau, -1)
        loss = -(target * log_q).sum(-1).mean()          # forward KL up to the teacher's entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
    return student


def main() -> None:
    seed = int(os.environ.get("SCHOLARLOOP_SEED", "0"))
    h = effective_hparams()
    model = train_model(seed, h)
    torch.save({"config": h, "state_dict": dict(model.state_dict())},
               os.environ["SCHOLARLOOP_ARTIFACT"])


if __name__ == "__main__":
    main()
