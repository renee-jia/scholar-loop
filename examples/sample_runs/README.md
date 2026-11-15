# Recorded runs

Complete runs of the whole loop, kept with their records. Start `python3 -m scholarloop.ui` from
the repository root and click **Explore a real run** to browse them; they open read-only.

| Folder | Question | What the registered prediction came to | Reviewers |
|---|---|---|---|
| [`distillation`](distillation/) | on-policy against off-policy distillation of a sequence model | falsified | 2.2 of 10 |
| [`agnews`](agnews/) | which regularizers help a small text classifier | partly held; the named comparator never ran | 2.7 of 10 |
| [`digits_skip`](digits_skip/) | does a skip connection change a shallow MLP's final error | neither supported nor falsified | 2.7 of 10 |
| [`digits`](digits/) | an earlier capture, before the treatment-versus-control rule existed | not registered | 2.2 of 10 |

The simulated reviewers rejected every paper, for reasons we agree with: one small task each, few
seeds, comparisons on validation data. The runs are here for their records, not their results.
