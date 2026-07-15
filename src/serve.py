"""FastAPI service: classify a support ticket, or abstain.

POST /classify with a message; get back the predicted intent, a confidence, and a
decision — `answer` when the calibrated confidence clears the frozen threshold, else
`abstain` (route to a human). The threshold is the operating point calibrated in Phase 3
(report/operating_point.json), never re-tuned here.

The model and retriever load once at startup (expensive on a laptop GPU); requests then
reuse them. Run with:  uvicorn src.serve:app
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import K_SHOTS, normalize
from src.data import label_maps, to_natural_language
from src.model import LabelScorer
from src.retrieval import ShotRetriever
from src.signals import logprob_signals

OPERATING_POINT = Path(__file__).resolve().parent.parent / "report" / "operating_point.json"

# Loaded at startup.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the operating point, the 4-bit model, and the retriever once."""
    op = json.loads(OPERATING_POINT.read_text())
    if op["signal"] != "logprob_margin":
        raise RuntimeError(f"serve.py assumes the logprob-margin signal, got {op['signal']}")
    canon, nl, _ = label_maps()
    _state.update(
        op=op,
        canon=canon,
        nl=nl,
        scorer=LabelScorer(),
        retriever=ShotRetriever(),
    )
    yield
    _state.clear()


app = FastAPI(title="LLM selective-prediction service", lifespan=lifespan)


class Ticket(BaseModel):
    message: str = Field(..., min_length=1, description="the customer-support message")


class Prediction(BaseModel):
    predicted_intent: str
    confidence: float = Field(..., description="max-softmax probability (interpretable)")
    margin: float = Field(..., description="calibrated signal: top prob minus runner-up")
    threshold: float = Field(..., description="frozen abstention threshold on the margin")
    decision: str = Field(..., description="'answer' or 'abstain'")
    guarantee: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "scorer" in _state}


@app.post("/classify", response_model=Prediction)
def classify(ticket: Ticket) -> Prediction:
    scorer: LabelScorer = _state["scorer"]
    retriever: ShotRetriever = _state["retriever"]
    canon, nl, op = _state["canon"], _state["nl"], _state["op"]

    shots = [(n.text, to_natural_language(n.label)) for n in retriever.retrieve(ticket.message, K_SHOTS)]
    scored = scorer.score_labels(ticket.message, nl, shots)
    mean_lp = normalize(scored.sum_logprobs, scored.n_tokens)

    predicted = canon[int(mean_lp.argmax())]
    conf, margin = logprob_signals(mean_lp)
    answer = margin >= op["tau"]

    return Prediction(
        predicted_intent=predicted,
        confidence=round(conf, 4),
        margin=round(margin, 4),
        threshold=round(op["tau"], 4),
        decision="answer" if answer else "abstain",
        guarantee=(
            f"On answered tickets, accuracy >= {op['target_accepted_accuracy']:.0%} "
            f"with probability >= {1 - op['delta']:.0%} (exchangeability assumed)."
        ),
    )
