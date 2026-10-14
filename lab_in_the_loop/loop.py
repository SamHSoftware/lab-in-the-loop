"""One lab-in-the-loop iteration: run the lab, update the model, propose the next run."""

from dataclasses import dataclass

import numpy as np

from lab_in_the_loop.simulator import N_COMPOUNDS, Lab, phenotype_loss
from lab_in_the_loop.surrogate import (
    Ensemble,
    propose_concentrations,
    sigmoid,
    train_ensemble,
)

# Experiment 0 is the current cells. Experiments 1 to 4 are a random screen.
N_SCREEN_EXPERIMENTS = 5
START_LOGIT_MEAN = -3.0
START_LOGIT_STD = 0.3
SCREEN_CONCENTRATION_RANGE = (0.05, 0.95)


@dataclass
class Measurement:
    """What the lab measured for the concentrations it just applied."""

    experiment_index: int
    phenotype: list[float]
    loss: float


@dataclass
class ModelSnapshot:
    """The model after it was trained on all experiments so far."""

    n_experiments: int
    fit_error: float
    input_hidden_weights: list[list[float]]
    hidden_output_weights: list[list[float]]


@dataclass
class Proposal:
    """The concentrations for the next iteration, and how they were chosen."""

    experiment_index: int
    method: str
    concentrations: list[float]
    predicted_phenotype: list[float] | None


@dataclass
class StepResult:
    """The outcome of one iteration, as the front end shows it."""

    measurement: Measurement | None
    model: ModelSnapshot | None
    proposal: Proposal


def choose_without_model(seed: int, experiment_index: int) -> np.ndarray:
    """Choose concentrations before there is enough data to trust the model.

    Experiment 0 uses concentrations near zero, which gives the current cells.
    The next experiments are a random screen, so the model sees varied data.

    Args:
        seed (int): Session seed.
        experiment_index (int): Position of this experiment in the session.

    Returns:
        concentrations (np.ndarray): Concentrations in (0, 1), shape (10,).
    """
    rng = np.random.default_rng([seed, experiment_index, 1])
    if experiment_index == 0:
        logits = rng.normal(
            loc=START_LOGIT_MEAN, scale=START_LOGIT_STD, size=N_COMPOUNDS
        )
        return sigmoid(logits=logits)
    return rng.uniform(*SCREEN_CONCENTRATION_RANGE, size=N_COMPOUNDS)


def snapshot_model(
    ensemble: Ensemble, concentrations: np.ndarray, phenotypes: np.ndarray
) -> ModelSnapshot:
    """Summarise the trained model for the network drawing on the web page.

    The weights are from the first net of the ensemble. The fit error uses the mean
    prediction of all nets on the training data.

    Args:
        ensemble (Ensemble): Trained nets.
        concentrations (np.ndarray): Training inputs, shape (n, 10).
        phenotypes (np.ndarray): Training targets, shape (n, 3).

    Returns:
        snapshot (ModelSnapshot): Fit error and the weights of the first net.
    """
    _, outputs = ensemble.forward(inputs=concentrations)
    mean_prediction = outputs.mean(axis=0)
    return ModelSnapshot(
        n_experiments=len(concentrations),
        fit_error=float(np.mean((mean_prediction - phenotypes) ** 2)),
        input_hidden_weights=ensemble.hidden_weights[0].tolist(),
        hidden_output_weights=ensemble.output_weights[0].tolist(),
    )


def run_step(seed: int, history: list[list[float]]) -> StepResult:
    """Run one lab-in-the-loop iteration.

    The last row of the history holds the concentrations to apply now. The lab
    measures them, the model is retrained on all experiments, and then the frozen
    model proposes the next concentrations. Nothing is stored between calls.

    Args:
        seed (int): Session seed, which fixes the hidden biology.
        history (list[list[float]]): Concentrations of every experiment, newest last.
            Empty at the start of a session.

    Returns:
        result (StepResult): The measurement, the updated model and the next proposal.
    """
    lab = Lab(seed=seed)
    next_index = len(history)
    measurement = None
    model = None

    if history:
        concentrations = np.array(history)
        phenotypes = np.array(
            [
                lab.run_experiment(concentrations=c, experiment_index=i)
                for i, c in enumerate(concentrations)
            ]
        )
        losses = [phenotype_loss(phenotype=p) for p in phenotypes]
        measurement = Measurement(
            experiment_index=next_index - 1,
            phenotype=phenotypes[-1].tolist(),
            loss=losses[-1],
        )
        ensemble = train_ensemble(
            concentrations=concentrations, phenotypes=phenotypes, seed=seed
        )
        model = snapshot_model(
            ensemble=ensemble, concentrations=concentrations, phenotypes=phenotypes
        )

    if next_index < N_SCREEN_EXPERIMENTS:
        proposal = Proposal(
            experiment_index=next_index,
            method="start" if next_index == 0 else "screen",
            concentrations=choose_without_model(
                seed=seed, experiment_index=next_index
            ).tolist(),
            predicted_phenotype=None,
        )
    else:
        next_concentrations, predicted = propose_concentrations(
            ensemble=ensemble,
            start_concentrations=concentrations[np.argmin(losses)],
            seed=seed * 1000 + next_index,
        )
        proposal = Proposal(
            experiment_index=next_index,
            method="model",
            concentrations=next_concentrations.tolist(),
            predicted_phenotype=predicted.tolist(),
        )

    return StepResult(measurement=measurement, model=model, proposal=proposal)
