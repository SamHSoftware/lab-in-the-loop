"""A fictitious wet lab which maps compound concentrations to cell phenotypes."""

import numpy as np

N_COMPOUNDS = 10
PHENOTYPE_NAMES = ("colour", "shape", "nucleus_size")
TARGET_PHENOTYPE = np.ones(len(PHENOTYPE_NAMES))

# Phenotype before tanh, with no compounds: blue (0), circular (-0.9), small (-0.9).
BASELINE_LATENT = np.array([0.0, -1.5, -1.5])

# Each compound raises, lowers or does not change each phenotype.
EFFECT_SIGNS = np.array([1.0, -1.0, 0.0])
EFFECT_SIGN_PROBABILITIES = np.array([0.375, 0.375, 0.25])
EFFECT_MAGNITUDE_RANGE = (0.5, 1.5)


class Lab:
    """The hidden ground truth which the model must learn through experiments.

    The seed fixes the effect of each compound on each phenotype, so one seed is one
    reproducible "biology". The model never sees the effects, only measurements.

    Args:
        seed (int): Seed which fixes the compound effects and the measurement noise.
        noise_std (float): Standard deviation of the measurement noise.
    """

    def __init__(self, seed: int, noise_std: float = 0.02) -> None:
        rng = np.random.default_rng(seed)
        effect_shape = (N_COMPOUNDS, len(PHENOTYPE_NAMES))
        signs = rng.choice(EFFECT_SIGNS, size=effect_shape, p=EFFECT_SIGN_PROBABILITIES)
        magnitudes = rng.uniform(*EFFECT_MAGNITUDE_RANGE, size=effect_shape)

        self.seed = seed
        self.noise_std = noise_std
        self.effects = signs * magnitudes

    def run_experiment(
        self, concentrations: np.ndarray, experiment_index: int
    ) -> np.ndarray:
        """Apply compounds to the cells and measure the phenotype.

        The noise depends on the experiment index, so a repeat of the same
        concentrations gives a slightly different measurement, as in a real lab.

        Args:
            concentrations (np.ndarray): Concentration of each compound, shape (10,).
            experiment_index (int): Position of this experiment in the session.

        Returns:
            phenotype (np.ndarray): Measured colour, shape and nucleus size in [-1, 1].
        """
        latent = BASELINE_LATENT + concentrations @ self.effects
        noise_rng = np.random.default_rng([self.seed, experiment_index])
        noise = noise_rng.normal(loc=0.0, scale=self.noise_std, size=latent.shape)
        return np.clip(np.tanh(latent) + noise, -1.0, 1.0)


def phenotype_loss(phenotype: np.ndarray) -> float:
    """Distance from the target phenotype, summed across the three phenotypes.

    The target is green, square cells with big nuclei, so every value is 1.

    Args:
        phenotype (np.ndarray): Colour, shape and nucleus size, shape (3,).

    Returns:
        loss (float): Sum of absolute distances to the target. 0 is perfect, 6 is worst.
    """
    return float(np.abs(TARGET_PHENOTYPE - phenotype).sum())
