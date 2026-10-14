"""Entry point which the web page calls through Pyodide, inside the browser."""

import json
from dataclasses import asdict

from lab_in_the_loop.loop import run_step


def step(seed: int, history_json: str) -> str:
    """Run one lab-in-the-loop iteration with JSON in and JSON out.

    JSON keeps the boundary between JavaScript and Python simple: one string each way.

    Args:
        seed (int): Session seed, which fixes the hidden biology.
        history_json (str): JSON list of the concentrations of every experiment,
            newest last. The newest row is applied in this iteration.

    Returns:
        result_json (str): The measurement, the updated model and the next proposal.
    """
    history = json.loads(history_json)
    return json.dumps(asdict(run_step(seed=seed, history=history)))
