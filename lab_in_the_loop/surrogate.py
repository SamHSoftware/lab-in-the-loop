"""The surrogate model: an ensemble of small feed-forward nets, and the proposal step.

The model is written in NumPy with hand-written backpropagation, so it runs in the
browser through Pyodide, which has NumPy but not PyTorch.
"""

from dataclasses import dataclass

import numpy as np

from lab_in_the_loop.simulator import N_COMPOUNDS, PHENOTYPE_NAMES

HIDDEN_SIZE = 32
N_OUTPUTS = len(PHENOTYPE_NAMES)

# Concentrations stay in [0.001, 0.999], so they never reach 0 or 1.
MAX_ABS_LOGIT = 6.9


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Map logits to values between 0 and 1.

    Args:
        logits (np.ndarray): Any real values.

    Returns:
        values (np.ndarray): Values in (0, 1), same shape as the input.
    """
    return 1.0 / (1.0 + np.exp(-logits))


def logit(values: np.ndarray) -> np.ndarray:
    """Inverse of the sigmoid, limited to the allowed logit range.

    Args:
        values (np.ndarray): Values in [0, 1].

    Returns:
        logits (np.ndarray): Logits in [-MAX_ABS_LOGIT, MAX_ABS_LOGIT].
    """
    limit = sigmoid(np.array(MAX_ABS_LOGIT))
    values = np.clip(values, 1.0 - limit, limit)
    return np.log(values / (1.0 - values))


@dataclass
class Ensemble:
    """The weights of several nets, stacked on the first axis (one row per net).

    Each net maps 10 concentrations to tanh(32 hidden units) to tanh(3 phenotypes).
    The tanh output keeps each prediction in [-1, 1], the same range as the lab.
    Weight matrices use the (out, in) layout, as in PyTorch.
    """

    hidden_weights: np.ndarray  # (members, hidden, inputs)
    hidden_bias: np.ndarray  # (members, hidden)
    output_weights: np.ndarray  # (members, outputs, hidden)
    output_bias: np.ndarray  # (members, outputs)

    def parameters(self) -> list[np.ndarray]:
        """List the trainable arrays, in the order which backward() uses.

        Returns:
            parameters (list[np.ndarray]): Weights and biases of all nets.
        """
        return [
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        ]

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the phenotype with every net.

        Args:
            inputs (np.ndarray): Concentrations, shape (n, 10).

        Returns:
            hidden (np.ndarray): Hidden activations, shape (members, n, 32).
            outputs (np.ndarray): Predicted phenotypes, shape (members, n, 3).
        """
        hidden = np.tanh(
            np.einsum("nd,mhd->mnh", inputs, self.hidden_weights)
            + self.hidden_bias[:, None, :]
        )
        outputs = np.tanh(
            np.einsum("mnh,moh->mno", hidden, self.output_weights)
            + self.output_bias[:, None, :]
        )
        return hidden, outputs

    def backward(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        outputs: np.ndarray,
        output_gradient: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Backpropagate a gradient from the outputs to the weights and the inputs.

        This is the chain rule, one layer at a time. The tanh derivative is 1 - tanh².

        Args:
            inputs (np.ndarray): Concentrations, shape (n, 10).
            hidden (np.ndarray): Hidden activations from forward().
            outputs (np.ndarray): Outputs from forward().
            output_gradient (np.ndarray): Gradient of the loss with respect to the
                outputs, shape (members, n, 3).

        Returns:
            parameter_gradients (list[np.ndarray]): One gradient per parameter.
            input_gradient (np.ndarray): Gradient for each input, shape (members, n, 10).
        """
        output_pre_gradient = output_gradient * (1.0 - outputs**2)
        output_weights_gradient = np.einsum("mno,mnh->moh", output_pre_gradient, hidden)
        output_bias_gradient = output_pre_gradient.sum(axis=1)

        hidden_gradient = np.einsum(
            "mno,moh->mnh", output_pre_gradient, self.output_weights
        )
        hidden_pre_gradient = hidden_gradient * (1.0 - hidden**2)
        hidden_weights_gradient = np.einsum("mnh,nd->mhd", hidden_pre_gradient, inputs)
        hidden_bias_gradient = hidden_pre_gradient.sum(axis=1)

        input_gradient = np.einsum(
            "mnh,mhd->mnd", hidden_pre_gradient, self.hidden_weights
        )
        parameter_gradients = [
            hidden_weights_gradient,
            hidden_bias_gradient,
            output_weights_gradient,
            output_bias_gradient,
        ]
        return parameter_gradients, input_gradient


class Adam:
    """The Adam optimiser with PyTorch defaults. It changes the arrays in place.

    Args:
        parameters (list[np.ndarray]): Arrays to optimise.
        learning_rate (float): Step size.
    """

    def __init__(
        self,
        parameters: list[np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moments = [np.zeros_like(p) for p in parameters]
        self.second_moments = [np.zeros_like(p) for p in parameters]
        self.step_count = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        """Take one optimisation step.

        Args:
            gradients (list[np.ndarray]): One gradient per parameter.
        """
        self.step_count += 1
        for parameter, gradient, first, second in zip(
            self.parameters, gradients, self.first_moments, self.second_moments
        ):
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient**2
            first_corrected = first / (1.0 - self.beta1**self.step_count)
            second_corrected = second / (1.0 - self.beta2**self.step_count)
            parameter -= (
                self.learning_rate
                * first_corrected
                / (np.sqrt(second_corrected) + self.epsilon)
            )


def init_ensemble(n_members: int, seed: int) -> Ensemble:
    """Create nets with random weights, as PyTorch's nn.Linear does.

    Weights and biases are uniform in ±1/sqrt(number of inputs to the layer).

    Args:
        n_members (int): Number of nets.
        seed (int): Seed for the random weights.

    Returns:
        ensemble (Ensemble): Untrained nets.
    """
    rng = np.random.default_rng(seed)
    hidden_bound = 1.0 / np.sqrt(N_COMPOUNDS)
    output_bound = 1.0 / np.sqrt(HIDDEN_SIZE)
    return Ensemble(
        hidden_weights=rng.uniform(
            -hidden_bound, hidden_bound, size=(n_members, HIDDEN_SIZE, N_COMPOUNDS)
        ),
        hidden_bias=rng.uniform(
            -hidden_bound, hidden_bound, size=(n_members, HIDDEN_SIZE)
        ),
        output_weights=rng.uniform(
            -output_bound, output_bound, size=(n_members, N_OUTPUTS, HIDDEN_SIZE)
        ),
        output_bias=rng.uniform(
            -output_bound, output_bound, size=(n_members, N_OUTPUTS)
        ),
    )


def train_ensemble(
    concentrations: np.ndarray,
    phenotypes: np.ndarray,
    seed: int,
    n_members: int = 5,
    n_epochs: int = 300,
    learning_rate: float = 0.01,
) -> Ensemble:
    """Train several nets from different random starts on all experiments so far.

    Where the nets disagree, the model is uncertain. The proposal step uses this to
    explore. Each call trains from zero, so the session needs no stored model.

    Args:
        concentrations (np.ndarray): Concentrations tested so far, shape (n, 10).
        phenotypes (np.ndarray): Measured phenotypes, shape (n, 3).
        seed (int): Seed for the weight initialisation.
        n_members (int): Number of nets in the ensemble.
        n_epochs (int): Number of full-batch training steps.
        learning_rate (float): Adam learning rate.

    Returns:
        ensemble (Ensemble): The trained nets.
    """
    ensemble = init_ensemble(n_members=n_members, seed=seed)
    optimiser = Adam(parameters=ensemble.parameters(), learning_rate=learning_rate)
    n_values = phenotypes.size

    for _ in range(n_epochs):
        hidden, outputs = ensemble.forward(inputs=concentrations)
        # Gradient of each net's mean squared error with respect to its outputs.
        output_gradient = 2.0 * (outputs - phenotypes[None]) / n_values
        gradients, _ = ensemble.backward(
            inputs=concentrations,
            hidden=hidden,
            outputs=outputs,
            output_gradient=output_gradient,
        )
        optimiser.step(gradients=gradients)
    return ensemble


def acquisition(
    ensemble: Ensemble, logits: np.ndarray, exploration_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score candidate experiments, and find the gradient of the score. Lower is better.

    The score is a lower confidence bound: the mean predicted loss minus a reward for
    ensemble disagreement (standard deviation). This makes the model optimistic where
    it is uncertain. The weights are frozen: only the logits get a gradient.

    Args:
        ensemble (Ensemble): Trained nets.
        logits (np.ndarray): Candidate concentrations as logits, shape (k, 10).
        exploration_weight (float): How much to reward ensemble disagreement.

    Returns:
        scores (np.ndarray): Score of each candidate, shape (k,).
        mean_prediction (np.ndarray): Mean predicted phenotype, shape (k, 3).
        logit_gradient (np.ndarray): Gradient of each score, shape (k, 10).
    """
    in_range = np.abs(logits) <= MAX_ABS_LOGIT
    concentrations = sigmoid(np.clip(logits, -MAX_ABS_LOGIT, MAX_ABS_LOGIT))
    hidden, outputs = ensemble.forward(inputs=concentrations)

    member_losses = np.abs(1.0 - outputs).sum(axis=2)
    n_members = member_losses.shape[0]
    mean_loss = member_losses.mean(axis=0)
    std_loss = member_losses.std(axis=0, ddof=1)
    scores = mean_loss - exploration_weight * std_loss

    # Chain rule: score -> member losses -> outputs -> concentrations -> logits.
    member_loss_gradient = 1.0 / n_members - exploration_weight * (
        member_losses - mean_loss
    ) / ((n_members - 1) * np.maximum(std_loss, 1e-12))
    output_gradient = member_loss_gradient[:, :, None] * np.sign(outputs - 1.0)
    _, input_gradient = ensemble.backward(
        inputs=concentrations,
        hidden=hidden,
        outputs=outputs,
        output_gradient=output_gradient,
    )
    sigmoid_slope = concentrations * (1.0 - concentrations)
    logit_gradient = input_gradient.sum(axis=0) * sigmoid_slope * in_range
    return scores, outputs.mean(axis=0), logit_gradient


def propose_concentrations(
    ensemble: Ensemble,
    start_concentrations: np.ndarray,
    seed: int,
    n_random_starts: int = 7,
    exploration_weight: float = 1.0,
    n_steps: int = 20,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Use gradient descent through the frozen ensemble to choose the next experiment.

    The weights are frozen and the concentrations are the trainable values. They are
    stored as logits and passed through a sigmoid, so a step changes the
    concentration less as it comes near 0 or 1. The search starts from the given
    point and from random points, so it can leave a local optimum.

    Args:
        ensemble (Ensemble): Trained nets.
        start_concentrations (np.ndarray): Start point, normally the best experiment.
        seed (int): Seed for the random start points.
        n_random_starts (int): Number of random start points.
        exploration_weight (float): How much to reward ensemble disagreement.
        n_steps (int): Number of gradient steps on the concentrations.
        learning_rate (float): Adam learning rate on the logits.

    Returns:
        concentrations (np.ndarray): Proposed concentrations, shape (10,).
        predicted_phenotype (np.ndarray): Mean ensemble prediction, shape (3,).
    """
    rng = np.random.default_rng(seed)
    starts = np.vstack(
        [start_concentrations, rng.uniform(size=(n_random_starts, N_COMPOUNDS))]
    )
    logits = logit(values=starts)
    optimiser = Adam(parameters=[logits], learning_rate=learning_rate)

    for _ in range(n_steps):
        _, _, logit_gradient = acquisition(
            ensemble=ensemble, logits=logits, exploration_weight=exploration_weight
        )
        optimiser.step(gradients=[logit_gradient])

    scores, mean_prediction, _ = acquisition(
        ensemble=ensemble, logits=logits, exploration_weight=exploration_weight
    )
    best = np.argmin(scores)
    best_logits = np.clip(logits[best], -MAX_ABS_LOGIT, MAX_ABS_LOGIT)
    return sigmoid(best_logits), mean_prediction[best]
