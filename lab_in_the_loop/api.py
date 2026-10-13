"""HTTP API which the web page calls. It is stateless: the browser keeps the session."""

import os
import threading
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lab_in_the_loop.loop import StepResult, run_step
from lab_in_the_loop.simulator import N_COMPOUNDS

MAX_EXPERIMENTS = 100
DEFAULT_ALLOWED_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"

Concentration = Annotated[float, Field(gt=0.0, lt=1.0)]
Concentrations = Annotated[
    list[Concentration], Field(min_length=N_COMPOUNDS, max_length=N_COMPOUNDS)
]


class StepRequest(BaseModel):
    """The session state which the browser sends with each iteration."""

    seed: int = Field(ge=0, le=2**31 - 1)
    history: list[Concentrations] = Field(max_length=MAX_EXPERIMENTS)


app = FastAPI(title="Lab in the loop")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# torch.manual_seed is global, so one step at a time keeps results reproducible.
_step_lock = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    """Tell the web page that the server is awake.

    Returns:
        status (dict[str, str]): Always {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/step")
def step(request: StepRequest) -> StepResult:
    """Run one lab-in-the-loop iteration.

    Args:
        request (StepRequest): Session seed and the concentrations of every
            experiment, newest last. The newest row is applied in this iteration.

    Returns:
        result (StepResult): The measurement, the updated model and the next proposal.
    """
    with _step_lock:
        return run_step(seed=request.seed, history=request.history)
