"""FastAPI evaluation endpoint for RAGVue."""

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .src.core.metrics_loader import load_metrics
from .src.core.manual_mode import evaluate as manual_evaluate
from .src.core.agentic_mode import AgenticOrchestrator

# ── Request / Response models ──

class EvalItem(BaseModel):
    question: str
    answer: str
    contexts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluateRequest(BaseModel):
    item: EvalItem
    metrics: Optional[List[str]] = None

class BatchEvaluateRequest(BaseModel):
    items: List[EvalItem]
    metrics: Optional[List[str]] = None

class AgenticEvaluateRequest(BaseModel):
    items: List[EvalItem]

class EvaluateResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    version: str
    metrics_count: int

class MetricsListResponse(BaseModel):
    metrics: List[str]
    count: int


# ── App factory ──

MAX_BATCH_ITEMS = 200

def create_app() -> FastAPI:
    app = FastAPI(title="RAGVue Evaluation API", version="0.2.0")
    registry = load_metrics()

    def _validate_metrics(requested: Optional[List[str]]) -> List[str]:
        if not requested:
            return list(registry.keys())
        unknown = [m for m in requested if m not in registry]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown metrics: {unknown}. Available: {sorted(registry.keys())}",
            )
        return requested

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version="0.2.0",
            metrics_count=len(registry),
        )

    @app.get("/metrics", response_model=MetricsListResponse)
    async def list_metrics():
        names = sorted(registry.keys())
        return MetricsListResponse(metrics=names, count=len(names))

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate_single(req: EvaluateRequest):
        metric_names = _validate_metrics(req.metrics)
        item_dict = req.item.model_dump()

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None, manual_evaluate, [item_dict], metric_names,
        )

        results = report.get("results", [])
        summary: Dict[str, Any] = {}
        for r in results:
            for m in r.get("metrics", []):
                name = m.get("name")
                if name:
                    summary[name] = m.get("score", 0.0)
        return EvaluateResponse(results=results, summary=summary)

    @app.post("/evaluate/batch", response_model=EvaluateResponse)
    async def evaluate_batch(req: BatchEvaluateRequest):
        if len(req.items) > MAX_BATCH_ITEMS:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(req.items)} exceeds limit of {MAX_BATCH_ITEMS}. Use checkpointing for large datasets.",
            )
        metric_names = _validate_metrics(req.metrics)
        items = [it.model_dump() for it in req.items]

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None, manual_evaluate, items, metric_names,
        )

        results = report.get("results", [])
        # aggregate summary: mean per metric
        agg: Dict[str, List[float]] = {}
        for r in results:
            for m in r.get("metrics", []):
                name = m.get("name")
                if name:
                    agg.setdefault(name, []).append(float(m.get("score", 0.0)))
        summary = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
        return EvaluateResponse(results=results, summary=summary)

    @app.post("/evaluate/agentic", response_model=EvaluateResponse)
    async def evaluate_agentic(req: AgenticEvaluateRequest):
        if len(req.items) > MAX_BATCH_ITEMS:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(req.items)} exceeds limit of {MAX_BATCH_ITEMS}. Use checkpointing for large datasets.",
            )
        items = [it.model_dump() for it in req.items]
        orch = AgenticOrchestrator()

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, orch.run, items)

        return EvaluateResponse(
            results=report.get("results", []),
            summary=report.get("summary", {}),
        )

    return app


app = create_app()


def serve():
    """Entry point for ``ragvue-api`` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="RAGVue Evaluation API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("ragvue.api:app", host=args.host, port=args.port, reload=args.reload)
