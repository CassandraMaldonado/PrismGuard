"""Synthetic labeled evaluation data for an AI product policy comparison."""

# I use synthetic data so the project is reproducible and does not expose real
# user conversations. The simulation includes task types, user cohorts, human
# quality labels, confidence scores, safety flags, latency, and inference cost.

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationSet:
    policy: np.ndarray
    task_type: np.ndarray
    cohort: np.ndarray
    human_quality: np.ndarray
    grounded: np.ndarray
    unsafe: np.ndarray
    confidence: np.ndarray
    latency_ms: np.ndarray
    cost_usd: np.ndarray
    reference_length: np.ndarray


TASKS = np.array(["research", "support", "coding", "creative"])
COHORTS = np.array(["new", "returning", "power"])


def simulate_evaluation(samples: int = 3_000, seed: int = 41) -> EvaluationSet:
    if samples < 400:
        raise ValueError("Use at least 400 samples for stable evaluation slices")
    rng = np.random.default_rng(seed)
    policy = rng.integers(0, 2, samples).astype(bool)
    task_type = rng.choice(TASKS, samples, p=[0.30, 0.30, 0.25, 0.15])
    cohort = rng.choice(COHORTS, samples, p=[0.32, 0.49, 0.19])
    task_base = np.select([task_type == "research", task_type == "support", task_type == "coding"], [0.69, 0.75, 0.67], default=0.73)
    candidate_lift = np.select([task_type == "research", task_type == "support", task_type == "coding"], [0.075, 0.032, 0.060], default=0.015)
    cohort_adjustment = np.select([cohort == "new", cohort == "power"], [-0.025, 0.020], default=0.0)
    quality_probability = np.clip(task_base + cohort_adjustment + policy * candidate_lift, 0.1, 0.97)
    human_quality = (rng.random(samples) < quality_probability).astype(float)
    grounded_probability = np.clip(0.72 + policy * 0.09 - (task_type == "creative") * 0.20, 0.05, 0.98)
    grounded = (rng.random(samples) < grounded_probability).astype(float)
    unsafe_probability = 0.010 + policy * 0.002 + (task_type == "creative") * 0.008
    unsafe = (rng.random(samples) < unsafe_probability).astype(float)
    # Candidate confidence is closer to human quality, but not perfectly calibrated.
    confidence = np.clip(0.43 + 0.42 * human_quality + policy * 0.025 + rng.normal(0, 0.11, samples), 0.01, 0.99)
    latency_ms = np.maximum(80, rng.lognormal(np.log(520 + policy * 130), 0.24, samples))
    cost_usd = np.maximum(0.001, 0.010 + policy * rng.gamma(2.2, 0.008, samples))
    reference_length = np.maximum(30, rng.normal(560 + (task_type == "research") * 220, 150, samples))
    return EvaluationSet(policy, task_type, cohort, human_quality, grounded, unsafe, confidence, latency_ms, cost_usd, reference_length)

