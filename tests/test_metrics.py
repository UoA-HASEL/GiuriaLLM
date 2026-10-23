"""Agreement metrics, checked against values computed by hand or by definition."""

from __future__ import annotations

import math

import pytest

from giuriallm.metaeval.metrics import (
    accuracy,
    cohen_kappa,
    distribution,
    earth_movers_distance,
    icc,
    jensen_shannon,
    kendall_tau,
    krippendorff_alpha,
    spearman,
)


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == pytest.approx(2 / 3)


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    # Complete disagreement on a balanced two-class problem: kappa = -1.
    assert cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)


def test_spearman_monotone_but_not_linear():
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    value = spearman([1, 1, 2, 3], [1, 1, 2, 3])
    assert value == pytest.approx(1.0)


def test_kendall_tau_known_value():
    # 5 items, one adjacent transposition: 9 concordant, 1 discordant -> 0.8.
    assert kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 5, 4]) == pytest.approx(0.8)


def test_icc_agreement_versus_disagreement():
    identical = icc([[4, 4], [3, 3], [5, 5], [2, 2]])
    scattered = icc([[4, 2], [3, 5], [5, 1], [2, 4]])
    assert identical == pytest.approx(1.0)
    assert scattered < identical


def test_krippendorff_alpha_perfect_agreement():
    assert krippendorff_alpha([[1, 1], [2, 2], [3, 3], [4, 4]]) == pytest.approx(1.0)


def test_krippendorff_alpha_is_near_zero_for_chance_agreement():
    units = [[1, 2], [2, 1], [1, 2], [2, 1]]
    value = krippendorff_alpha(units, level="nominal")
    assert value is not None and value < 0.0


def test_krippendorff_alpha_tolerates_missing_ratings():
    assert krippendorff_alpha([[1, 1], [2, None], [3, 3]]) is not None


def test_distribution_normalises_over_support():
    assert distribution([1, 1, 2], [1, 2, 3]) == pytest.approx([2 / 3, 1 / 3, 0.0])


def test_emd_is_zero_for_identical_distributions():
    p = [0.2, 0.3, 0.5]
    assert earth_movers_distance(p, p, [1, 2, 3]) == pytest.approx(0.0)


def test_emd_grows_with_distance_between_distributions():
    support = [1, 2, 3, 4, 5]
    near = earth_movers_distance([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], support)
    far = earth_movers_distance([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], support)
    assert near == pytest.approx(0.25)
    assert far == pytest.approx(1.0)
    assert far > near


def test_emd_separates_what_a_correlation_cannot():
    """Two judges with the same rank order but different magnitudes of error."""
    support = [1, 2, 3, 4, 5]
    human = distribution([3, 3, 3], support)
    close = distribution([2, 3, 4], support)
    distant = distribution([1, 1, 5], support)
    assert earth_movers_distance(close, human, support) < earth_movers_distance(
        distant, human, support
    )


def test_jensen_shannon_bounds():
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert jensen_shannon(p, p) == pytest.approx(0.0)
    assert jensen_shannon(p, q) == pytest.approx(1.0)
    assert 0.0 <= jensen_shannon([0.5, 0.5], q) <= 1.0


def test_jensen_shannon_is_symmetric():
    p, q = [0.7, 0.2, 0.1], [0.2, 0.5, 0.3]
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))


def test_metrics_return_none_rather_than_raising_on_degenerate_input():
    assert spearman([1], [1]) is None
    assert kendall_tau([], []) is None
    assert icc([[1, 1]]) is None
    assert krippendorff_alpha([]) is None
    assert not math.isnan(jensen_shannon([0.5, 0.5], [0.5, 0.5]) or 0.0)
