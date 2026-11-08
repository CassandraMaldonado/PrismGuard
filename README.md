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
