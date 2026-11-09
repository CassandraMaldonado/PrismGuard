"""FastAPI delivery layer for PrismGuard."""

from functools import lru_cache
from fastapi import FastAPI, Query
from .analysis import analyze

app = FastAPI(title="PrismGuard API", version="0.1.0")

@lru_cache(maxsize=32)
def _brief(seed: int, samples: int):
    return analyze(seed, samples)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/brief")
def brief(seed: int = Query(41, ge=0, le=1_000_000), samples: int = Query(3_000, ge=400, le=100_000)):
    return _brief(seed, samples)
