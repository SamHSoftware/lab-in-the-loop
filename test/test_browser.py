"""Tests for the entry point which the web page calls through Pyodide."""

import json

from lab_in_the_loop.browser import step


def test_step_round_trip() -> None:
    """Tests a start call and one iteration with JSON in and JSON out."""
    start = json.loads(step(seed=7, history_json="[]"))
    assert start["measurement"] is None
    assert start["proposal"]["method"] == "start"

    history = json.dumps([start["proposal"]["concentrations"]])
    result = json.loads(step(seed=7, history_json=history))
    assert result["measurement"]["experiment_index"] == 0
    assert len(result["model"]["input_hidden_weights"]) == 32
