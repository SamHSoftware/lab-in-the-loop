"""Tests for one lab-in-the-loop iteration."""

import numpy as np

from lab_in_the_loop.loop import N_SCREEN_EXPERIMENTS, StepResult, run_step


def run_session(seed: int, n_iterations: int) -> list[StepResult]:
    """Start a session and run several iterations, as the browser does.

    Args:
        seed (int): Session seed.
        n_iterations (int): Number of iterations after the start call.

    Returns:
        results (list[StepResult]): The start result, then one result per iteration.
    """
    history: list[list[float]] = []
    results = [run_step(seed=seed, history=history)]
    for _ in range(n_iterations):
        history.append(results[-1].proposal.concentrations)
        results.append(run_step(seed=seed, history=history))
    return results


def test_start_call_only_proposes() -> None:
    """Tests that an empty history gives start concentrations and nothing else."""
    result = run_step(seed=1, history=[])
    assert result.measurement is None
    assert result.model is None
    assert result.proposal.method == "start"
    assert max(result.proposal.concentrations) < 0.2


def test_iteration_measures_then_updates_model() -> None:
    """Tests that an iteration measures the newest row and retrains on all rows."""
    results = run_session(seed=1, n_iterations=2)
    assert results[2].measurement is not None
    assert results[2].model is not None
    assert results[2].measurement.experiment_index == 1
    assert results[2].model.n_experiments == 2
    assert np.array(results[2].model.input_hidden_weights).shape == (32, 10)
    assert np.array(results[2].model.hidden_output_weights).shape == (3, 32)


def test_proposal_methods() -> None:
    """Tests that proposals start, screen, and then use the model."""
    results = run_session(seed=1, n_iterations=N_SCREEN_EXPERIMENTS)
    methods = [result.proposal.method for result in results]
    assert methods == ["start"] + ["screen"] * (N_SCREEN_EXPERIMENTS - 1) + ["model"]
    assert results[-2].proposal.predicted_phenotype is None
    assert results[-1].proposal.predicted_phenotype is not None


def test_step_is_reproducible() -> None:
    """Tests that the same seed and history give the same result."""
    results = run_session(seed=2, n_iterations=6)
    history = [result.proposal.concentrations for result in results[:-1]]
    assert run_step(seed=2, history=history) == run_step(seed=2, history=history)


def test_loop_improves_on_start() -> None:
    """Tests that the loop finds better cells than the start cells."""
    results = run_session(seed=2, n_iterations=15)
    losses = [result.measurement.loss for result in results if result.measurement]
    assert min(losses) < losses[0] - 2.0
