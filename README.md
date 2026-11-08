# PrismGuard

**A practical way to evaluate an AI feature before rolling it out.**

I built PrismGuard because I do not think a higher quality score alone is enough to justify launching an AI feature. Before I would roll something out, I would also want to understand whether it is safe, whether its confidence scores make sense, how much slower or more expensive it is, and whether it works equally well for different types of users and tasks.

## The question

In this project, a new retrieval and response policy is being considered as a replacement for the existing system. Before shipping it, I wanted to answer a few practical questions:

- Does it improve task quality and groundedness?
- Does it create new safety or fairness concerns?
- Are its confidence scores calibrated enough for users to trust?
- Is the quality gain worth the latency and inference cost?
- Does it hold up across task types and user cohorts?

PrismGuard simulates a labeled evaluation set and compares the candidate policy with the current one. It then looks at the quality and reliability results together, checks whether the evaluation traffic has shifted, and produces a clear `SHIP`, `ITERATE`, or `HOLD` recommendation.

## What makes this project different

- Looks at quality, safety, latency, and cost together instead of focusing on one score.
- Compares the candidate policy with the existing one using confidence intervals and task-level results.
- Checks whether the model's confidence actually matches human-rated quality.
- Looks for traffic drift before relying too heavily on offline evaluation results.
- Shows where the candidate performs differently across task types and user groups.
- Uses clear launch criteria, so it is easy to see why the recommendation is to ship, iterate, or hold.
- Includes a Streamlit dashboard, tested analysis code, and an API for exploring the results.


## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run streamlit_app.py
```

You can also print a reproducible decision brief:

```bash
python -m prism_guard.cli --seed 41 --samples 3000
```

## How it works

```mermaid
flowchart LR
    A["Create labeled evaluation data"] --> B["Check coverage and traffic drift"]
    B --> C["Compare the current and candidate policies"]
    C --> D["Review quality safety latency and cost"]
    D --> E["Check calibration and performance by group"]
    E --> F["Make a rollout recommendation"]
```

