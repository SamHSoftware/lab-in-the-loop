"""The surrogate model: a small ensemble of feed-forward nets, and the proposal step."""

import numpy as np
import torch
from torch import nn

from lab_in_the_loop.simulator import N_COMPOUNDS, PHENOTYPE_NAMES

# Concentrations stay in [0.001, 0.999], so they never reach 0 or 1.
MAX_ABS_LOGIT = 6.9


class SurrogateNet(nn.Module):
    """Feed-forward net which predicts the phenotype from compound concentrations.

    The tanh output keeps the prediction in [-1, 1], the same range as the lab.

    Args:
        hidden_size (int): Number of units in the hidden layer.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_COMPOUNDS, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(PHENOTYPE_NAMES)),
            nn.Tanh(),
        )

    def forward(self, concentrations: torch.Tensor) -> torch.Tensor:
        """Predict the phenotype.

        Args:
            concentrations (torch.Tensor): Shape (batch, 10).

        Returns:
            phenotype (torch.Tensor): Shape (batch, 3).
        """
        return self.layers(concentrations)


def train_ensemble(
    concentrations: np.ndarray,
    phenotypes: np.ndarray,
    seed: int,
    n_members: int = 5,
    n_epochs: int = 300,
    learning_rate: float = 0.01,
) -> list[SurrogateNet]:
    """Train several nets from different random starts on all experiments so far.

    Where the nets disagree, the model is uncertain. The proposal step uses this to
    explore. Each call trains from zero, so the server keeps no state.

    Args:
        concentrations (np.ndarray): Concentrations tested so far, shape (n, 10).
        phenotypes (np.ndarray): Measured phenotypes, shape (n, 3).
        seed (int): Seed for the weight initialisation.
        n_members (int): Number of nets in the ensemble.
        n_epochs (int): Number of full-batch training steps per net.
        learning_rate (float): Adam learning rate.

    Returns:
        ensemble (list[SurrogateNet]): The trained nets.
    """
    inputs = torch.tensor(concentrations, dtype=torch.float32)
    targets = torch.tensor(phenotypes, dtype=torch.float32)

    ensemble = []
    for member_index in range(n_members):
        torch.manual_seed(seed * n_members + member_index)
        net = SurrogateNet()
        optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for _ in range(n_epochs):
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(net(inputs), targets)
            loss.backward()
            optimiser.step()
        ensemble.append(net)
    return ensemble


def acquisition(
    ensemble: list[SurrogateNet], logits: torch.Tensor, exploration_weight: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidate experiments. Lower is better.

    The score is a lower confidence bound: the mean predicted loss minus a reward for
    ensemble disagreement. This makes the model optimistic where it is uncertain.

    Args:
        ensemble (list[SurrogateNet]): Trained nets.
        logits (torch.Tensor): Candidate concentrations as logits, shape (n, 10).
        exploration_weight (float): How much to reward ensemble disagreement.

    Returns:
        scores (torch.Tensor): Score of each candidate, shape (n,).
        mean_prediction (torch.Tensor): Mean predicted phenotype, shape (n, 3).
    """
    concentrations = torch.sigmoid(logits.clamp(-MAX_ABS_LOGIT, MAX_ABS_LOGIT))
    predictions = torch.stack([net(concentrations) for net in ensemble])
    member_losses = (1.0 - predictions).abs().sum(dim=2)
    scores = member_losses.mean(dim=0) - exploration_weight * member_losses.std(dim=0)
    return scores, predictions.mean(dim=0)


def propose_concentrations(
    ensemble: list[SurrogateNet],
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
        ensemble (list[SurrogateNet]): Trained nets.
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
    for net in ensemble:
        net.requires_grad_(False)

    generator = torch.Generator().manual_seed(seed)
    starts = torch.cat(
        [
            torch.tensor(start_concentrations, dtype=torch.float32)[None],
            torch.rand(n_random_starts, N_COMPOUNDS, generator=generator),
        ]
    )
    logits = torch.logit(starts).clamp(-MAX_ABS_LOGIT, MAX_ABS_LOGIT).requires_grad_()
    optimiser = torch.optim.Adam([logits], lr=learning_rate)

    for _ in range(n_steps):
        optimiser.zero_grad()
        scores, _ = acquisition(
            ensemble=ensemble, logits=logits, exploration_weight=exploration_weight
        )
        scores.sum().backward()
        optimiser.step()

    with torch.no_grad():
        scores, mean_predictions = acquisition(
            ensemble=ensemble, logits=logits, exploration_weight=exploration_weight
        )
        best = scores.argmin()
        concentrations = torch.sigmoid(
            logits[best].clamp(-MAX_ABS_LOGIT, MAX_ABS_LOGIT)
        )
    return concentrations.numpy(), mean_predictions[best].numpy()
