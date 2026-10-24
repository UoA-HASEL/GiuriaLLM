"""Agreement metrics, implemented without a numerical stack.

Two families, deliberately separated:

*Classic* metrics (accuracy, Cohen's kappa, Krippendorff's alpha, Spearman,
Kendall, ICC) compare the judge to a single collapsed human label. They are what
most LLM-as-a-Judge studies report.

*Distribution-aware* metrics (earth mover's distance, Jensen-Shannon divergence)
compare the judge's distribution of verdicts against the *distribution* of expert
opinion, without collapsing disagreement among experts into a gold label. These
address the objection that single-gold-label statistics misrepresent human-vs-LLM
comparison when the experts themselves disagree.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from statistics import mean

Number = float


# --------------------------------------------------------------------------
# Classic, single-gold-label metrics
# --------------------------------------------------------------------------


def accuracy(a: Sequence, b: Sequence) -> float | None:
    pairs = list(zip(a, b, strict=True))
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohen_kappa(a: Sequence, b: Sequence) -> float | None:
    """Chance-corrected agreement between two raters on categorical labels."""
    pairs = list(zip(a, b, strict=True))
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    count_a = Counter(x for x, _ in pairs)
    count_b = Counter(y for _, y in pairs)
    expected = sum((count_a[k] / n) * (count_b[k] / n) for k in set(count_a) | set(count_b))
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _ranks(values: Sequence[Number]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(a: Sequence[Number], b: Sequence[Number]) -> float | None:
    pairs = list(zip(a, b, strict=True))
    n = len(pairs)
    if n < 2:
        return None
    mean_a, mean_b = mean(x for x, _ in pairs), mean(y for _, y in pairs)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in pairs)
    var_a = sum((x - mean_a) ** 2 for x, _ in pairs)
    var_b = sum((y - mean_b) ** 2 for _, y in pairs)
    if var_a == 0 or var_b == 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def spearman(a: Sequence[Number], b: Sequence[Number]) -> float | None:
    """Rank correlation, with mid-ranks for ties."""
    if len(a) < 2:
        return None
    return pearson(_ranks(a), _ranks(b))


def kendall_tau(a: Sequence[Number], b: Sequence[Number]) -> float | None:
    """Kendall's tau-b (tie-corrected)."""
    n = len(a)
    if n < 2:
        return None
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            product = da * db
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if da == 0:
                    ties_a += 1
                if db == 0:
                    ties_b += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def icc(matrix: Sequence[Sequence[Number]]) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    ``matrix`` is targets x raters (e.g. items x [judge, human]).
    """
    rows = [list(r) for r in matrix if r]
    n = len(rows)
    if n < 2:
        return None
    k = len(rows[0])
    if k < 2 or any(len(r) != k for r in rows):
        return None

    grand = mean(v for row in rows for v in row)
    row_means = [mean(row) for row in rows]
    col_means = [mean(row[c] for row in rows) for c in range(k)]

    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((v - grand) ** 2 for row in rows for v in row)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        return None
    return (ms_rows - ms_error) / denominator


def krippendorff_alpha(units: Sequence[Sequence], level: str = "interval") -> float | None:
    """Krippendorff's alpha over units x ratings, with missing ratings allowed.

    ``level`` is ``"interval"`` (numeric) or ``"nominal"`` (categorical).
    """
    usable = [[v for v in unit if v is not None] for unit in units]
    usable = [unit for unit in usable if len(unit) >= 2]
    if not usable:
        return None

    coincidence: Counter[tuple] = Counter()
    for unit in usable:
        m = len(unit)
        for i, c in enumerate(unit):
            for j, k in enumerate(unit):
                if i == j:
                    continue
                coincidence[(c, k)] += 1.0 / (m - 1)

    values = sorted({v for unit in usable for v in unit}, key=_sort_key)
    marginals = {c: sum(coincidence[(c, k)] for k in values) for c in values}
    n = sum(marginals.values())
    if n <= 1:
        return None

    def delta(c, k) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        return (float(c) - float(k)) ** 2

    observed = sum(coincidence[(c, k)] * delta(c, k) for c in values for k in values) / n
    expected = sum(marginals[c] * marginals[k] * delta(c, k) for c in values for k in values) / (
        n * (n - 1)
    )
    if expected == 0:
        return None
    return 1.0 - observed / expected


def _sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


# --------------------------------------------------------------------------
# Distribution-aware metrics
# --------------------------------------------------------------------------


def distribution(values: Sequence, support: Sequence) -> list[float]:
    """Empirical distribution of ``values`` over an ordered ``support``."""
    counts = Counter(values)
    total = sum(counts[v] for v in support)
    if total == 0:
        return [0.0] * len(support)
    return [counts[v] / total for v in support]


def earth_movers_distance(
    p: Sequence[float], q: Sequence[float], support: Sequence | None = None
) -> float | None:
    """1-D Wasserstein distance between two distributions on an ordinal support.

    Normalised to [0, 1] by the support's range, so values are comparable across
    scales. Unlike a correlation, it respects *how far apart* disagreements are.
    """
    if len(p) != len(q) or not p:
        return None
    if support is not None and len(support) == len(p):
        try:
            positions = [float(s) for s in support]
        except (TypeError, ValueError):
            positions = [float(i) for i in range(len(p))]
    else:
        positions = [float(i) for i in range(len(p))]

    total = 0.0
    carried = 0.0
    for i in range(len(p) - 1):
        carried += p[i] - q[i]
        total += abs(carried) * (positions[i + 1] - positions[i])
    span = positions[-1] - positions[0]
    return total / span if span > 0 else total


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float | None:
    """Jensen-Shannon divergence in bits, in [0, 1]. Symmetric, always finite."""
    if len(p) != len(q) or not p:
        return None
    m = [(a + b) / 2.0 for a, b in zip(p, q, strict=True)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    total = 0.0
    for a, b in zip(p, q, strict=True):
        if a > 0 and b > 0:
            total += a * math.log2(a / b)
    return total
