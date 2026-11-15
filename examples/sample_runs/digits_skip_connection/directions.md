# Research directions — residual connections versus parameter count in shallow MLP image classifiers

## Your steer

> What the author wants to find out: I want to know whether a residual (skip) path helps a shallow MLP image classifier on a small dataset because of the path itself, or only because the variant has more parameters or trains differently.
> The author's hunches:
>   H1. On a network this shallow the residual path adds little once parameter count and training steps are matched.
> Stay away from: Regularization studies; Pretrained encoders or external data; Pure hyperparameter sweeps with no mechanism behind them
> Kind of result the author values: A two-arm comparison with a parameter-matched control; a clean negative result is welcome.
> The author already knows or has tried: A logistic-regression baseline reaches about 3.8% error on this split

## Direction map

_Directions are lines of inquiry, not experiments. Choose one or more; ideas are generated only inside the directions you choose._

| # | direction | stance toward your hunches | builds on |
|---|---|---|---|
| 1 | ✅ **Separate the effect of a residual path from the effect of added...** | yours | — |

### 1. Separate the effect of a residual path from the effect of added...  ✅ chosen

Separate the effect of a residual path from the effect of added parameters in a shallow MLP, using a parameter-matched control under identical training resources

- Written by you. Not yet tied to reviewed papers.
- **ID:** `dir_0369ab623f`
