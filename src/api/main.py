"""FastAPI server for Ripple: Real-time risk signals and health endpoints."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.scoring.store import get_signal_by_id, init_db, query_signals

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is created on startup."""
    init_db()
    yield


app = FastAPI(
    title="Ripple Risk Engine API",
    version="1.0.0",
    description="Real-time financial risk detection, event clustering, impact scoring, and signals API.",
    lifespan=lifespan,
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint confirming engine status."""
    return {
        "status": "ok",
        "service": "ripple-risk-engine",
        "version": "1.0.0",
    }


@app.get("/signals", tags=["Signals"])
def list_signals(
    since: Optional[str] = Query(None, description="ISO timestamp filter (e.g. 2026-10-10T00:00:00)"),
    event_type: Optional[str] = Query(None, description="Event type category filter"),
    min_impact: Optional[float] = Query(None, ge=1.0, le=10.0, description="Minimum impact score threshold (1.0 to 10.0)"),
    entity: Optional[str] = Query(None, description="Entity name, ticker, sector, or country substring"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of signals to return"),
) -> list[dict[str, Any]]:
    """Retrieve filtered list of risk signals ordered by most recent first."""
    signals = query_signals(
        since=since,
        event_type=event_type,
        min_impact=min_impact,
        entity=entity,
        limit=limit,
    )
    return [s.to_dict() for s in signals]


@app.get("/signals/{id}", tags=["Signals"])
def get_signal(id: str) -> dict[str, Any]:
    """Retrieve a specific signal by its unique event ID."""
    sig = get_signal_by_id(id)
    if sig is None:
        raise HTTPException(status_code=404, detail=f"Signal with ID '{id}' not found.")
    return sig.to_dict()
