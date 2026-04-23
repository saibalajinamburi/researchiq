import os
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.inference import ResearchIQInference


APP_VERSION = os.getenv("RESEARCHIQ_VERSION", "0.1.0")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RESEARCHIQ_RATE_LIMIT_PER_MINUTE", "60"))
MAX_BATCH_SIZE = int(os.getenv("RESEARCHIQ_MAX_BATCH_SIZE", "16"))

REQUESTS = Counter(
    "researchiq_requests_total",
    "Total API requests.",
    ["endpoint", "method", "status"],
)
PREDICTIONS = Counter(
    "researchiq_predictions_total",
    "Total model predictions.",
    ["category", "cached"],
)
LATENCY = Histogram(
    "researchiq_prediction_latency_seconds",
    "Prediction latency in seconds.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
CACHE_SIZE = Gauge("researchiq_cache_size", "Number of in-process cached predictions.")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=12000)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    label_id: int
    category: str
    confidence: float
    top_categories: list[dict[str, Any]]
    cached: bool


inference = ResearchIQInference()
rate_window: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(
    title="ResearchIQ API",
    version=APP_VERSION,
    description="Scientific paper category classifier using ONNX embeddings and ONNX classifier inference.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RESEARCHIQ_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        endpoint = request.url.path
        REQUESTS.labels(endpoint=endpoint, method=request.method, status=status).inc()
        if endpoint in {"/predict", "/predict/batch"}:
            LATENCY.observe(time.perf_counter() - started)


def client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    now = time.time()
    key = client_id(request)
    window = rate_window[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    window.append(now)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "ResearchIQ",
        "version": APP_VERSION,
        "links": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "model": "/model/info",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_exists = inference.classifier_path.exists()
    metadata_exists = inference.metadata_path.exists()
    return {
        "status": "ok" if model_exists and metadata_exists else "degraded",
        "model_exists": model_exists,
        "metadata_exists": metadata_exists,
        "version": APP_VERSION,
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    return inference.info()


@app.post("/predict", response_model=Prediction)
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    enforce_rate_limit(request)
    result = inference.predict(payload.text)
    PREDICTIONS.labels(category=result["category"], cached=str(result["cached"]).lower()).inc()
    CACHE_SIZE.set(inference.cache.stats()["size"])
    return result


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(payload: BatchPredictRequest, request: Request) -> list[dict[str, Any]]:
    enforce_rate_limit(request)
    if any(len(text) < 20 for text in payload.texts):
        raise HTTPException(status_code=422, detail="Every text must contain at least 20 characters.")
    results = inference.predict_batch(payload.texts)
    for result in results:
        PREDICTIONS.labels(category=result["category"], cached=str(result["cached"]).lower()).inc()
    CACHE_SIZE.set(inference.cache.stats()["size"])
    return results


@app.get("/metrics")
def metrics() -> Response:
    CACHE_SIZE.set(inference.cache.stats()["size"])
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
