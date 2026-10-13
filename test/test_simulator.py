"""Tests for the fictitious lab."""

import numpy as np

from lab_in_the_loop.simulator import N_COMPOUNDS, Lab, phenotype_loss


def test_same_seed_gives_same_biology() -> None:
    """Tests that a seed fixes the compound effects."""
    assert np.array_equal(Lab(seed=3).effects, Lab(seed=3).effects)
    assert not np.array_equal(Lab(seed=3).effects, Lab(seed=4).effects)


def test_about_a_quarter_of_effects_are_zero() -> None:
    """Tests the 25% chance that a compound does not change a phenotype."""
    effects = np.concatenate([Lab(seed=seed).effects.ravel() for seed in range(500)])
    assert abs(np.mean(effects == 0.0) - 0.25) < 0.02


def test_no_compounds_gives_current_cells() -> None:
    """Tests that near-zero concentrations give blue, circular cells with small nuclei."""
    lab = Lab(seed=0, noise_std=0.0)
    phenotype = lab.run_experiment(
        concentrations=np.zeros(N_COMPOUNDS), experiment_index=0
    )
    assert np.allclose(phenotype, [0.0, -0.905, -0.905], atol=0.001)


def test_phenotype_stays_in_range() -> None:
    """Tests that the phenotype is in [-1, 1] for extreme concentrations."""
    lab = Lab(seed=0)
    for concentrations in (np.zeros(N_COMPOUNDS), np.ones(N_COMPOUNDS)):
        phenotype = lab.run_experiment(
            concentrations=concentrations, experiment_index=1
        )
        assert np.all(np.abs(phenotype) <= 1.0)


def test_phenotype_loss() -> None:
    """Tests the loss at the target and at the opposite corner."""
    assert phenotype_loss(phenotype=np.ones(3)) == 0.0
    assert phenotype_loss(phenotype=-np.ones(3)) == 6.0
