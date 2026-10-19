import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayes_cash.strategy import wasserstein_distance


def test_wasserstein_two_uses_squared_transport_on_unequal_grid():
    theta = np.array([0.0, 0.2, 1.0])
    beliefs = np.array([0.5, 0.5, 0.0])
    reference = np.array([0.0, 0.25, 0.75])
    # Move 0.25 from 0 to 0.2, 0.25 from 0 to 1, and 0.5 from 0.2 to 1.
    squared_cost = 0.25 * 0.2**2 + 0.25 * 1.0**2 + 0.5 * 0.8**2
    assert_allclose(wasserstein_distance(beliefs, reference, theta), np.sqrt(squared_cost))


def test_wasserstein_two_differs_from_mass_times_distance():
    theta = np.array([0.0, 0.4, 1.0])
    beliefs = np.array([0.75, 0.0, 0.25])
    reference = np.array([1.0, 0.0, 0.0])
    # A quarter of the mass travels one unit: W2 = sqrt(0.25), whereas W1 = 0.25.
    assert_allclose(wasserstein_distance(beliefs, reference, theta), 0.5)


@pytest.mark.parametrize("source, target", [(0, 0), (0, 1), (1, 3), (3, 0)])
def test_point_mass_distance_is_hypothesis_separation(source, target):
    theta = np.array([0.0, 0.1, 0.5, 1.0])
    point_masses = np.eye(theta.size)
    distance = wasserstein_distance(point_masses[source], point_masses[target], theta)
    assert_allclose(distance, abs(theta[source] - theta[target]))


def test_equal_means_do_not_hide_different_belief_shapes():
    theta = np.array([0.0, 0.5, 1.0])
    beliefs = np.array([0.5, 0.0, 0.5])
    reference = np.array([0.0, 1.0, 0.0])
    assert_allclose(beliefs @ theta, reference @ theta)
    assert_allclose(wasserstein_distance(beliefs, reference, theta), 0.5)


def test_distance_broadcasts_over_rounds_and_agents_and_is_symmetric():
    theta = np.array([0.0, 0.25, 1.0])
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    beliefs = np.array([reference, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    expected = np.array([[0.0, 0.0], [0.25, 0.75]])
    assert_allclose(wasserstein_distance(beliefs, reference, theta), expected)
    assert_allclose(wasserstein_distance(reference, beliefs, theta), expected)


def test_zero_probability_bins_preserve_transport_symmetry_and_identity():
    theta = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
    beliefs = np.array([0.25, 0.0, 0.0, 0.75, 0.0])
    reference = np.array([0.0, 0.5, 0.0, 0.0, 0.5])
    # Move 0.25 from 0 to 0.2, 0.25 from 0.7 to 0.2, and 0.5 from 0.7 to 1.
    expected = np.sqrt(0.25 * 0.2**2 + 0.25 * 0.5**2 + 0.5 * 0.3**2)
    assert_allclose(wasserstein_distance(beliefs, reference, theta), expected)
    assert_allclose(wasserstein_distance(reference, beliefs, theta), expected)
    assert_allclose(wasserstein_distance(beliefs, beliefs, theta), 0.0, atol=1e-15)


def test_normalized_pmf_roundoff_does_not_create_spurious_transport():
    theta = np.linspace(0.0, 1.0, 51)
    beliefs = np.arange(1, theta.size + 1, dtype=float)
    beliefs /= beliefs.sum()
    reference = np.zeros_like(beliefs)
    reference[25] = 1.0
    # Transport to a point mass is exactly its root mean squared displacement.
    expected = np.sqrt(np.sum(beliefs * (theta - theta[25])**2))
    assert_allclose(wasserstein_distance(beliefs, reference, theta), expected, rtol=1e-14)
    assert_allclose(wasserstein_distance(reference, beliefs, theta), expected, rtol=1e-14)
    assert_allclose(wasserstein_distance(beliefs, beliefs, theta), 0.0, atol=1e-15)
