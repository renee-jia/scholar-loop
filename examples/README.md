# Examples

Run every script from the repository root.

## Free: no API key, scripted model

| Script | What it shows | Needs |
|---|---|---|
| `quickstart.py` | one idea through the three-tier funnel on the stub engine | `.[dev]` |
| `ideation_demo.py` | literature review, candidate ideas, a human pick, then the loop | `.[dev]` |
| `governed_campaign.py` | a parallel population funnel under a governor that decides when to stop | `.[dev]` |
| `writing_demo.py` | outline, panel discussion, author sign-off, key points, draft, review | `.[dev]` |
| `campaign_demo.py` | the full agent chain on real torch training runs | `.[dev,engines]` |
| `compare_search.py` | agent loop against random search at the same budget (`--arms random` is free) | `.[dev,engines]` |

## Live: uses the Anthropic API and costs money

Set `ANTHROPIC_API_KEY` and install `.[llm]`. Set `SCHOLARLOOP_SPEND_CAP` (USD) to bound the
spend: it is checked before every request.

| Script | What it does |
|---|---|
| `e2e_live.py` | the full agent chain against the real API |
| `run_to_paper.py` | a complete flow, campaign then staged write-up, saving every artifact |
| `case_study_agnews.py` | the AG News case study under one shared dollar cap, with the human checkpoints; `case_study_agnews.sh` calls it |

## Inputs

`steer.json` is an author's steer, `research_brief.json` a detailed brief, and `human_idea.json`
an idea supplied by the author instead of generated.

## Recorded runs

[`sample_runs/`](sample_runs/) holds four complete runs with their records. Start
`python3 -m scholarloop.ui` and click **Explore a real run** to browse them read-only.
