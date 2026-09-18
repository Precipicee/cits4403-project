"""Statistics for hypothesis decision rules: bootstrap intervals, break-point fits, BIC."""
from __future__ import annotations

import numpy as np
import pandas as pd

from source.metrics import bootstrap_ci, spearman  # noqa: F401  (re-exported for notebooks)

N_BOOT = 2000


def mean_ci(values, level=0.95, rng=None):
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(values, N_BOOT, level, rng)
    return float(np.nanmean(values)), float(lo), float(hi)


def paired_difference(a, b, level=0.95, rng=None):
    """Mean of ``b - a`` over seeds paired by repeat, with a bootstrap interval."""
    return mean_ci(np.asarray(b, float) - np.asarray(a, float), level, rng)


def linear_rss(x, y):
    coef = np.polyfit(x, y, 1)
    return float(np.sum((y - np.polyval(coef, x)) ** 2)), coef


def segmented_fit(x, y, grid=np.arange(0.05, 0.9501, 0.01)):
    """Continuous two-segment linear fit, break-point by grid search."""
    best = None
    for b in grid:
        design = np.column_stack([np.ones_like(x), x, np.maximum(x - b, 0)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(np.sum((y - design @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(b), coef)
    rss, b, coef = best
    return {"rss": rss, "break": b, "slope_below": float(coef[1]), "slope_above": float(coef[1] + coef[2])}


def bic(rss, n, p):
    return n * np.log(rss / n) + p * np.log(n)


def h1_threshold(runs: pd.DataFrame, metric="mean_wealth", rng=None):
    rng = rng if rng is not None else np.random.default_rng(1)
    x = runs["population.giver_fraction"].to_numpy(float)
    y = runs[metric].to_numpy(float)
    n = len(x)
    rss_lin, _ = linear_rss(x, y)
    seg = segmented_fit(x, y)
    delta = float(bic(rss_lin, n, 2) - bic(seg["rss"], n, 4))
    repeats = runs["repeat"].to_numpy()
    uniq = np.unique(repeats)
    boot_break, boot_diff = [], []
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([np.flatnonzero(repeats == r) for r in pick])
        s = segmented_fit(x[idx], y[idx])
        boot_break.append(s["break"])
        boot_diff.append(s["slope_above"] - s["slope_below"])
    diff_lo, diff_hi = np.quantile(boot_diff, [0.025, 0.975])
    verdict = ("supported" if delta > 10 and diff_lo > 0 else "refuted" if delta <= 2 else "inconclusive")
    return {"delta_bic": delta, "break": seg["break"], "break_ci": tuple(np.quantile(boot_break, [0.025, 0.975])),
            "slope_below": seg["slope_below"], "slope_above": seg["slope_above"],
            "slope_diff_ci": (float(diff_lo), float(diff_hi)), "verdict": verdict}


def _paired(runs, f_a, f_b, metric):
    a = runs[np.isclose(runs["population.giver_fraction"], f_a)].sort_values("repeat")[metric].to_numpy(float)
    b = runs[np.isclose(runs["population.giver_fraction"], f_b)].sort_values("repeat")[metric].to_numpy(float)
    return a, b


def h2_decoupling(runs: pd.DataFrame, rng=None):
    fractions = np.round(np.sort(runs["population.giver_fraction"].unique()), 2)
    pairs = []
    for fa, fb in zip(fractions[:-1], fractions[1:]):
        ga, gb = _paired(runs, fa, fb, "gini")
        wa, wb = _paired(runs, fa, fb, "mean_wealth")
        upper = fa >= 0.5
        level = 0.99 if upper else 0.995
        dg = paired_difference(ga, gb, level, rng)
        dw = paired_difference(wa, wb, level, rng)
        dg_full = paired_difference(ga, gb, 0.995, rng)
        pairs.append({"f": fa, "f_next": fb, "upper": upper, "dG": dg[0], "dG_lo": dg[1], "dG_hi": dg[2],
                      "dW": dw[0], "dW_lo": dw[1], "dW_hi": dw[2], "dG_full_lo": dg_full[1]})
    table = pd.DataFrame(pairs)
    upper = table[table["upper"]]
    supported = bool(((upper["dG_lo"] > 0) & (upper["dW_lo"] > 0)).any())
    x = runs["population.giver_fraction"].to_numpy(float)
    g = runs["gini"].to_numpy(float)
    ok = np.isfinite(g)
    rho = spearman(x[ok], g[ok])
    rng = rng if rng is not None else np.random.default_rng(2)
    boots = [spearman(x[ok][i], g[ok][i]) for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(500))]
    rho_hi = float(np.nanquantile(boots, 0.975))
    refuted = bool(not (table["dG_full_lo"] > 0).any() and rho_hi < 0)
    verdict = "supported" if supported else "refuted" if refuted else "inconclusive"
    return {"pairs": table, "spearman": rho, "spearman_hi": rho_hi, "verdict": verdict}


def h3_segregation(runs: pd.DataFrame, fractions=(0.3, 0.4, 0.5, 0.6, 0.7), rng=None):
    sub = runs[np.isclose(runs["population.giver_fraction"].to_numpy()[:, None], fractions).any(axis=1)]
    slope = mean_ci(sub["assortativity_slope"], rng=rng)
    degree = mean_ci(sub["realised_degree_taker"] - sub["realised_degree_giver"], rng=rng)
    weight = mean_ci(sub["weight_taker"] - sub["weight_giver"], rng=rng)
    supported = slope[1] > 0 and degree[2] < 0 and weight[2] < 0
    refuted = slope[1] <= 0 and degree[1] <= 0 <= degree[2] and weight[1] <= 0 <= weight[2]
    return {"slope": slope, "degree_gap": degree, "weight_gap": weight,
            "verdict": "supported" if supported else "refuted" if refuted else "inconclusive"}


def eta_squared(values, groups) -> float:
    values, groups = np.asarray(values, float), np.asarray(groups)
    grand = values.mean()
    between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2 for g in np.unique(groups))
    total = ((values - grand) ** 2).sum()
    return float(between / total) if total > 0 else float("nan")


def h4_equal_poverty(runs: pd.DataFrame, rng=None):
    rng = rng if rng is not None else np.random.default_rng(3)
    f = runs["population.giver_fraction"].to_numpy(float)
    taker, giver = runs[f <= 0.3 + 1e-9], runs[f >= 0.7 - 1e-9]
    med = mean_ci(taker.groupby("repeat")["median_wealth"].mean().to_numpy() -
                  giver.groupby("repeat")["median_wealth"].mean().to_numpy(), rng=rng)
    gin = mean_ci(taker.groupby("repeat")["gini"].mean().to_numpy() -
                  giver.groupby("repeat")["gini"].mean().to_numpy(), rng=rng)
    repeats = runs["repeat"].unique()
    diffs = []
    for _ in range(500):
        pick = rng.choice(repeats, len(repeats))
        sample = pd.concat([runs[runs["repeat"] == r] for r in pick])
        diffs.append(eta_squared(sample["median_wealth"], sample["population.giver_fraction"]) -
                     eta_squared(sample["gini"].fillna(0), sample["population.giver_fraction"]))
    eta_diff = (float(np.mean(diffs)), *map(float, np.quantile(diffs, [0.025, 0.975])))
    supported = med[2] < 0 and gin[0] <= 0 and gin[2] < 0.05 and eta_diff[1] > 0
    refuted = med[2] >= 0 or gin[1] > 0
    return {"median_gap": med, "gini_gap": gin, "eta_sq_diff": eta_diff,
            "verdict": "supported" if supported else "refuted" if refuted else "inconclusive"}


CALIBRATION_AXES = ("encounter.eps", "ties.decay_rate", "ties.severance_threshold", "encounter.max_received")
CALIBRATION_START = {"encounter.eps": 0.1, "ties.decay_rate": 0.05, "ties.severance_threshold": 0.1,
                     "encounter.max_received": 4}


def drift_ci(slopes, rng=None):
    """Mean per-seed slope with its 95% bootstrap interval."""
    return mean_ci(np.asarray(slopes, float), rng=rng)


def calibration_table(runs: pd.DataFrame, start=CALIBRATION_START, coop=(0.2, 0.8), turnover=(0.001, 0.2),
                      component=0.95, steady=0.9, steady_rule="cell", gini_tol=1e-4, income_tol=1e-3,
                      rng=None) -> pd.DataFrame:
    """One row per setting with the pass criteria and the summed relative change from ``start``.

    ``steady_rule="cell"`` (default) passes a setting when the 95% intervals of the mean per-seed slopes
    of income Gini and relative mean income lie inside the tolerances, which separates drift shared by
    all seeds from seed-level noise. ``steady_rule="seeds"`` passes it when at least ``steady`` of its seeds pass the
    per-run test."""
    rng = rng if rng is not None else np.random.default_rng(4)
    axes = list(start)
    table = runs.groupby(axes).agg(coop_rate=("coop_rate", "mean"), turnover=("turnover", "mean"),
                                   largest_component=("largest_component", "mean"),
                                   steady_share=("steady_state", "mean")).reset_index()
    table["pass_coop"] = table["coop_rate"].between(*coop)
    table["pass_turnover"] = table["turnover"].between(*turnover)
    table["pass_component"] = table["largest_component"] >= component
    if steady_rule == "seeds":
        table["pass_steady"] = table["steady_share"] >= steady
    elif steady_rule == "cell":
        inside = []
        for _, group in runs.groupby(axes):
            _, g_lo, g_hi = drift_ci(group["slope_income_gini"], rng)
            _, m_lo, m_hi = drift_ci(group["slope_mean_income"], rng)
            inside.append(-gini_tol < g_lo and g_hi < gini_tol and -income_tol < m_lo and m_hi < income_tol)
        table["pass_steady"] = inside
    else:
        raise ValueError(f"unknown steady_rule {steady_rule!r}")
    table["passes"] = table[["pass_coop", "pass_turnover", "pass_component", "pass_steady"]].all(axis=1)
    table["distance"] = sum((table[a] - v).abs() / v for a, v in start.items())
    return table.sort_values(["passes", "distance"], ascending=[False, True]).reset_index(drop=True)


def select_calibration(table: pd.DataFrame, start=CALIBRATION_START) -> dict | None:
    """Passing setting closest to the starting values; ties go to the earlier row of the grid order."""
    passing = table[table["passes"]]
    if passing.empty:
        return None
    best = passing[np.isclose(passing["distance"], passing["distance"].min())].sort_values(list(start)).iloc[0]
    return {a: (int(best[a]) if isinstance(v, int) else float(best[a])) for a, v in start.items()}
