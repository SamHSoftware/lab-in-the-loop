"""Tests for the hand-written backpropagation, the optimiser and the proposal step."""

from typing import Callable

import numpy as np

from lab_in_the_loop.simulator import Lab
from lab_in_the_loop.surrogate import (
    Adam,
    acquisition,
    init_ensemble,
    propose_concentrations,
    train_ensemble,
)


def numerical_gradient(function: Callable[[], float], array: np.ndarray) -> np.ndarray:
    """Estimate a gradient with central finite differences.

    Each value of the array is moved a little up and down, in place, and back.

    Args:
        function (Callable[[], float]): Returns the scalar to differentiate.
        array (np.ndarray): The array to differentiate with respect to.

    Returns:
        gradient (np.ndarray): Estimated gradient, same shape as the array.
    """
    step = 1e-6
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        upper = function()
        array[index] = original - step
        lower = function()
        array[index] = original
        gradient[index] = (upper - lower) / (2 * step)
    return gradient


def test_backward_matches_finite_differences() -> None:
    """Tests the weight gradients of the mean squared error."""
    rng = np.random.default_rng(0)
    ensemble = init_ensemble(n_members=2, seed=0)
    inputs = rng.uniform(size=(4, 10))
    targets = rng.uniform(-1, 1, size=(4, 3))

    def loss() -> float:
        _, outputs = ensemble.forward(inputs=inputs)
        return float(((outputs - targets[None]) ** 2).mean(axis=(1, 2)).sum())

    hidden, outputs = ensemble.forward(inputs=inputs)
    gradients, _ = ensemble.backward(
        inputs=inputs,
        hidden=hidden,
        outputs=outputs,
        output_gradient=2.0 * (outputs - targets[None]) / targets.size,
    )
    for parameter, gradient in zip(ensemble.parameters(), gradients):
        expected = numerical_gradient(function=loss, array=parameter)
        assert np.allclose(gradient, expected, atol=1e-7)


def test_acquisition_gradient_matches_finite_differences() -> None:
    """Tests the gradient of the score with respect to the concentration logits."""
    rng = np.random.default_rng(1)
    ensemble = init_ensemble(n_members=5, seed=1)
    logits = rng.normal(size=(3, 10))

    def total_score() -> float:
        scores, _, _ = acquisition(
            ensemble=ensemble, logits=logits, exploration_weight=1.0
        )
        return float(scores.sum())

    _, _, gradient = acquisition(
        ensemble=ensemble, logits=logits, exploration_weight=1.0
    )
    expected = numerical_gradient(function=total_score, array=logits)
    assert np.allclose(gradient, expected, atol=1e-6)


def test_adam_finds_minimum_of_a_bowl() -> None:
    """Tests the optimiser on a simple quadratic."""
    values = np.array([3.0, -2.0])
    optimiser = Adam(parameters=[values], learning_rate=0.1)
    for _ in range(500):
        optimiser.step(gradients=[2.0 * values])
    assert np.allclose(values, 0.0, atol=1e-2)


def test_training_fits_the_data() -> None:
    """Tests that the trained ensemble predicts its training data well."""
    lab = Lab(seed=0)
    rng = np.random.default_rng(0)
    concentrations = rng.uniform(size=(20, 10))
    phenotypes = np.array(
        [
            lab.run_experiment(concentrations=c, experiment_index=i)
            for i, c in enumerate(concentrations)
        ]
    )
    ensemble = train_ensemble(
        concentrations=concentrations, phenotypes=phenotypes, seed=0
    )
    _, outputs = ensemble.forward(inputs=concentrations)
    assert np.mean((outputs.mean(axis=0) - phenotypes) ** 2) < 0.01


def test_proposal_never_reaches_zero_or_one() -> None:
    """Tests the sigmoid bounds with an optimiser which pushes very hard."""
    ensemble = init_ensemble(n_members=5, seed=0)
    proposal, _ = propose_concentrations(
        ensemble=ensemble,
        start_concentrations=np.full(10, 0.5),
        seed=0,
        n_steps=500,
        learning_rate=1.0,
    )
    assert np.all(proposal > 0.0) and np.all(proposal < 1.0)
