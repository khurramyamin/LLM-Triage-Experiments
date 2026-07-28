#!/usr/bin/env python3
"""
generic_roc.py — ROC + utility recovery for any labeled beliefs/decisions dataset.

This is a portable, self-contained version of the analysis used in the Nature
Medicine commentary (see the repository root README). It takes a *generic*
labeled dataset — one row per case, with an elicited **belief** (a probability
in [0, 1]), a ground-truth **label**, and optionally an observed binary
**decision** — and does four things:

  1. Builds the belief-vs-label **ROC curve** and reports the AUROC.

  2. **Backs out the FN/FP cost ratio implied by the dataset** using the
     revealed-preference discrete-choice (logit) fit of Yamin et al. If a
     ``decision`` column is present, the ratio is recovered from the observed
     decisions relative to the beliefs (the priority the decision-maker behaved
     *as if* it held). If no decisions are given, the ratio is recovered from
     the ground-truth labels themselves.

  3. Lets you specify a **target cost ratio** (FN:FP) to evaluate by.

  4. Finds the **best fixed utility ratio** for that target cost ratio: the
     single belief threshold that minimises the target-weighted expected cost
     on the ROC, expressed as the FN/FP cost ratio you would prompt an LLM with
     to reproduce that operating point on unseen data.

The r = 1 (1:1) special case is exactly the accuracy-maximising "Best Fixed
Utility" operating point plotted in the family-specific
``nature_medicine_paper/figures/fig3_roc_grid_*.png`` panels.

Cost-ratio convention
---------------------
A cost ratio is always **FN/FP** (false-negative cost divided by false-positive
cost), matching the rest of this repository. On the command line you may pass:

  * a plain number, e.g. ``--cost-ratio 1``   -> FN/FP = 1   (accuracy)
  * a ratio ``FN:FP``, e.g. ``--cost-ratio 10:1`` -> FN/FP = 10 (safety-leaning)
  * ``--cost-ratio 1:5``                       -> FN/FP = 0.2 (resource-leaning)

Usage
-----
    python generic_roc.py --input example_dataset.csv --cost-ratio 1
    python generic_roc.py --input mydata.csv \
        --belief-col p --label-col y --decision-col action --cost-ratio 5:1 \
        --output-dir out

Input CSV schema (column names are configurable via CLI flags)
--------------------------------------------------------------
    belief   : float in [0, 1]   — P(positive/needs action | context)   [required]
    label    : binary            — ground-truth outcome (1 = positive)   [required]
    decision : binary            — observed action taken (1 = act)       [optional]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TRUE_TOKENS = {"1", "1.0", "yes", "y", "true", "t", "positive", "pos", "admit"}
_FALSE_TOKENS = {"0", "0.0", "no", "n", "false", "f", "negative", "neg", "discharge"}


def parse_cost_ratio(text: str | float) -> float:
    """Parse a target cost ratio into a single FN/FP float.

    Accepts a plain number (interpreted directly as FN/FP) or an ``FN:FP``
    string such as ``"10:1"`` or ``"1:5"``.
    """
    s = str(text).strip()
    if ":" in s:
        left, right = s.split(":", 1)
        fn = float(left)
        fp = float(right)
        if fp <= 0 or fn < 0:
            raise ValueError(f"invalid cost ratio {text!r}: costs must be positive")
        return fn / fp
    val = float(s)
    if val <= 0:
        raise ValueError(f"invalid cost ratio {text!r}: must be > 0")
    return val


def _to_binary(raw: Optional[str], *, field_name: str) -> Optional[int]:
    """Coerce a cell to 0/1. Returns None for empty cells."""
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token == "":
        return None
    if token in _TRUE_TOKENS:
        return 1
    if token in _FALSE_TOKENS:
        return 0
    # Fall back to numeric coercion (e.g. "1", "0", "1.0").
    try:
        num = float(token)
    except ValueError as exc:  # noqa: TRY003
        raise ValueError(
            f"cannot interpret {raw!r} in column {field_name!r} as binary 0/1"
        ) from exc
    if num == 1:
        return 1
    if num == 0:
        return 0
    raise ValueError(
        f"value {raw!r} in column {field_name!r} is not binary (expected 0 or 1)"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    beliefs: np.ndarray                      # float, in [0, 1]
    labels: np.ndarray                       # int, {0, 1}
    decisions: Optional[np.ndarray] = None   # int, {0, 1} or None
    n_rows_read: int = 0
    n_rows_used: int = 0


def load_dataset(
    path: Path,
    *,
    belief_col: str = "belief",
    label_col: str = "label",
    decision_col: Optional[str] = None,
) -> Dataset:
    """Load a labeled beliefs/decisions dataset from a CSV file.

    Rows with a missing belief or label are skipped. If ``decision_col`` is
    given, rows missing the decision are still kept (decision recorded as NaN)
    but such rows are dropped from the revealed-preference fit.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty")
        missing = [c for c in (belief_col, label_col) if c not in reader.fieldnames]
        if decision_col and decision_col not in reader.fieldnames:
            missing.append(decision_col)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        beliefs: list[float] = []
        labels: list[int] = []
        decisions: list[float] = []
        n_read = 0
        for row in reader:
            n_read += 1
            b_raw = (row.get(belief_col) or "").strip()
            y = _to_binary(row.get(label_col), field_name=label_col)
            if b_raw == "" or y is None:
                continue
            belief = float(b_raw)
            if not (0.0 <= belief <= 1.0):
                raise ValueError(
                    f"belief {belief!r} in column {belief_col!r} is outside [0, 1]"
                )
            beliefs.append(belief)
            labels.append(y)
            if decision_col:
                d = _to_binary(row.get(decision_col), field_name=decision_col)
                decisions.append(float("nan") if d is None else float(d))

    ds = Dataset(
        beliefs=np.asarray(beliefs, dtype=float),
        labels=np.asarray(labels, dtype=int),
        decisions=(np.asarray(decisions, dtype=float) if decision_col else None),
        n_rows_read=n_read,
        n_rows_used=len(beliefs),
    )
    if ds.n_rows_used == 0:
        raise ValueError(f"{path} has no usable rows (belief + label required)")
    return ds


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def _trapezoid(y, x) -> float:
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(fn(y, x))


def roc_curve(scores, labels):
    """ROC (fpr, tpr, thresholds) and AUROC from scores vs binary labels.

    Sweeps the decision threshold from high to low so the curve runs from
    (0, 0) to (1, 1); AUROC is the trapezoidal area. No sklearn dependency.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return (np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([np.inf, -np.inf]), float("nan"))
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    thr = np.concatenate([[np.inf], s_sorted])
    auc = _trapezoid(tpr, fpr)
    return fpr, tpr, thr, auc


# ---------------------------------------------------------------------------
# Revealed-preference cost-function fit (back out the implied FN/FP ratio)
# ---------------------------------------------------------------------------

def fit_cost_function(beliefs, decisions) -> dict:
    """Fit a binary logit cost function to (belief, decision) pairs.

    Model (Yamin et al., action set {act, not-act}, beta = 1):

        EL(act     | p) = c_FP * (1 - p)
        EL(not-act | p) = c_FN * p
        Pr(act | p)     = sigmoid( c_FN * p - c_FP * (1 - p) )

    Systematic costs c = (c_FP, c_FN) are estimated by maximum likelihood
    (scipy L-BFGS-B). The interpretable quantity is the ratio c_FN / c_FP.
    """
    from scipy.optimize import minimize

    p = np.asarray(beliefs, dtype=float)
    a = np.asarray(decisions, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(a)
    p, a = p[mask], a[mask]
    n = int(len(a))
    if n == 0:
        return {"n": 0, "c_fp": None, "c_fn": None, "ratio_fn_fp": None,
                "loglik": None, "frac_act": None, "degenerate": True}

    frac_act = float(a.mean())
    if frac_act in (0.0, 1.0):
        # An all-act or all-not-act vector cannot identify finite costs.
        return {"n": n, "c_fp": None, "c_fn": None, "ratio_fn_fp": None,
                "loglik": None, "frac_act": frac_act, "degenerate": True}

    def neg_loglik(c: np.ndarray) -> float:
        c_fp, c_fn = c
        z = c_fn * p - c_fp * (1.0 - p)
        log_p_act = -np.logaddexp(0.0, -z)
        log_p_noact = -np.logaddexp(0.0, z)
        ll = a * log_p_act + (1.0 - a) * log_p_noact
        return -float(ll.sum())

    res = minimize(
        neg_loglik, x0=np.array([1.0, 1.0]), method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )
    c_fp, c_fn = float(res.x[0]), float(res.x[1])
    return {
        "n": n,
        "c_fp": c_fp,
        "c_fn": c_fn,
        "ratio_fn_fp": (c_fn / c_fp) if c_fp > 0 else None,
        "loglik": -float(res.fun),
        "frac_act": frac_act,
        "degenerate": False,
    }


def back_out_ratio(ds: Dataset) -> dict:
    """Back out the FN/FP ratio implied by a dataset.

    Uses the observed decisions if present, otherwise the ground-truth labels.
    The returned ``source`` field records which was used.
    """
    if ds.decisions is not None and np.isfinite(ds.decisions).any():
        fit = fit_cost_function(ds.beliefs, ds.decisions)
        fit["source"] = "decisions"
    else:
        fit = fit_cost_function(ds.beliefs, ds.labels.astype(float))
        fit["source"] = "labels"
    return fit


# ---------------------------------------------------------------------------
# Best fixed utility for a target cost ratio
# ---------------------------------------------------------------------------

@dataclass
class BestFixedUtility:
    cost_ratio_fn_fp: float       # target FN/FP the operating point is chosen for
    threshold: float              # belief threshold (act iff belief >= threshold)
    utility_ratio_fn_fp: float    # FN/FP to prompt the LLM with -> reproduces threshold
    tpr: float
    fpr: float
    accuracy: float
    total_cost: float             # target-weighted cost at this threshold (c_FP = 1)
    tp: int
    fp: int
    tn: int
    fn: int


def best_fixed_utility(beliefs, labels, cost_ratio: float = 1.0) -> BestFixedUtility:
    """Find the belief threshold minimising the target-weighted expected cost.

    For a target cost ratio r = c_FN / c_FP (with c_FP fixed to 1), the cost at
    threshold ``thr`` (rule: act iff belief >= thr) is ``FP(thr) + r * FN(thr)``.
    The threshold that minimises this is the best operating point reachable on
    the belief ROC for that priority.

    The **utility ratio** returned is ``(1 - thr) / thr`` — the FN/FP cost ratio
    that, given to an LLM as a cost function, implies the Bayes-optimal referral
    threshold ``p* = c_FP / (c_FP + c_FN) = thr``. This is what you prompt with
    on unseen data. It equals ``cost_ratio`` only when the beliefs are
    well-calibrated at the relevant operating point.

    With ``cost_ratio = 1`` this reduces to the accuracy-maximising threshold —
    the "Best Fixed Utility" point in the family-specific ROC grids.
    """
    p = np.asarray(beliefs, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = int(len(y))
    r = float(cost_ratio)
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)

    candidates = np.unique(np.concatenate([[0.0], p, [1.0 + 1e-9]]))
    best: Optional[BestFixedUtility] = None
    best_cost = np.inf
    for thr in candidates:
        pred = (p >= thr).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        cost = fp + r * fn  # c_FP = 1, c_FN = r
        # Strict improvement keeps the smallest threshold among ties, matching
        # analyze_sweep.best_fixed_threshold.
        if cost < best_cost - 1e-12:
            best_cost = cost
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            util = (1.0 - thr) / thr if thr > 0 else float("inf")
            best = BestFixedUtility(
                cost_ratio_fn_fp=r,
                threshold=float(thr),
                utility_ratio_fn_fp=float(util),
                tpr=float(tp / P),
                fpr=float(fp / N),
                accuracy=float((tp + tn) / n),
                total_cost=float(cost),
                tp=tp, fp=fp, tn=tn, fn=fn,
            )
    assert best is not None
    return best


def operating_point(decisions, labels):
    """(fpr, tpr, accuracy) of a set of observed binary decisions vs labels."""
    d = np.asarray(decisions, dtype=float)
    y = np.asarray(labels, dtype=int)
    mask = ~np.isnan(d)
    d = d[mask].astype(int)
    y = y[mask]
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    tpr = float(((d == 1) & (y == 1)).sum() / P)
    fpr = float(((d == 1) & (y == 0)).sum() / N)
    acc = float((d == y).mean())
    return fpr, tpr, acc


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_roc(ds: Dataset, bfu: BestFixedUtility, auc: float, out_path: Path,
             *, implied: Optional[dict] = None, title: str = "Belief ROC") -> bool:
    """Draw the belief ROC with the best-fixed-utility operating point.

    Returns True if the figure was written, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(skipping plot: {exc})", file=sys.stderr)
        return False

    fpr, tpr, _thr, _auc = roc_curve(ds.beliefs, ds.labels)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(fpr, tpr, "-", color="0.45", lw=2.6, zorder=2,
            label=f"Belief ROC (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, zorder=1, label="Chance")

    # Observed decisions' operating point (if provided).
    if ds.decisions is not None and np.isfinite(ds.decisions).any():
        dfpr, dtpr, _ = operating_point(ds.decisions, ds.labels)
        if not np.isnan(dfpr):
            lbl = "Observed decisions"
            if implied and implied.get("ratio_fn_fp"):
                lbl += f" (implied FN/FP={implied['ratio_fn_fp']:.2f})"
            ax.plot(dfpr, dtpr, "o", markersize=13, color="#3182bd",
                    markeredgecolor="k", markeredgewidth=0.8, zorder=5, label=lbl)

    # Best fixed utility operating point for the target cost ratio.
    ax.plot(bfu.fpr, bfu.tpr, "s", color="gold", markersize=18,
            markeredgecolor="k", markeredgewidth=1.0, zorder=6,
            label=(f"Best Fixed Utility @ FN/FP={_fmt_ratio(bfu.cost_ratio_fn_fp)}\n"
                   f"(prompt FN/FP={_fmt_ratio(bfu.utility_ratio_fn_fp)}, "
                   f"thr={bfu.threshold:.2f})"))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.93)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------

def _fmt_ratio(r: float) -> str:
    """Human-readable FN:FP ratio string, e.g. 0.2 -> '1:5', 10 -> '10:1'."""
    if r is None or not np.isfinite(r):
        return "inf"
    if r >= 1:
        return f"{r:.3g}:1"
    return f"1:{(1.0 / r):.3g}"


def analyze(ds: Dataset, cost_ratio: float) -> dict:
    """Run the full analysis and return a JSON-serialisable results dict."""
    _fpr, _tpr, _thr, auc = roc_curve(ds.beliefs, ds.labels)
    implied = back_out_ratio(ds)
    bfu = best_fixed_utility(ds.beliefs, ds.labels, cost_ratio)
    return {
        "n_rows_read": ds.n_rows_read,
        "n_rows_used": ds.n_rows_used,
        "n_positive": int((ds.labels == 1).sum()),
        "n_negative": int((ds.labels == 0).sum()),
        "auroc": auc,
        "implied_ratio": implied,
        "target_cost_ratio_fn_fp": cost_ratio,
        "best_fixed_utility": asdict(bfu),
    }


def print_report(results: dict) -> None:
    r = results
    imp = r["implied_ratio"]
    bfu = r["best_fixed_utility"]
    print("=" * 70)
    print("Generic ROC / utility analysis")
    print("=" * 70)
    print(f"rows used            : {r['n_rows_used']} of {r['n_rows_read']} "
          f"({r['n_positive']} positive, {r['n_negative']} negative)")
    print(f"AUROC (belief rank)  : {r['auroc']:.3f}")
    print("-" * 70)
    print("Implied FN/FP ratio (backed out of the dataset)")
    if imp.get("degenerate"):
        print(f"  source={imp['source']}: degenerate (all-act or all-not-act); "
              "cannot identify a finite ratio")
    else:
        print(f"  source = {imp['source']}   (n={imp['n']})")
        print(f"  c_FP={imp['c_fp']:.3f}  c_FN={imp['c_fn']:.3f}  "
              f"FN/FP={imp['ratio_fn_fp']:.3f}  ({_fmt_ratio(imp['ratio_fn_fp'])})")
    print("-" * 70)
    print(f"Target cost ratio (evaluate by): FN/FP={r['target_cost_ratio_fn_fp']:.3g} "
          f"({_fmt_ratio(r['target_cost_ratio_fn_fp'])})")
    print("Best fixed utility for that target:")
    print(f"  belief threshold      : {bfu['threshold']:.3f}  (act iff belief >= thr)")
    print(f"  operating point       : TPR={bfu['tpr']:.3f}  FPR={bfu['fpr']:.3f}  "
          f"accuracy={bfu['accuracy']:.3f}")
    print(f"  best fixed UTILITY    : FN/FP={bfu['utility_ratio_fn_fp']:.3g} "
          f"({_fmt_ratio(bfu['utility_ratio_fn_fp'])})")
    print("  --> prompt the LLM with this FN/FP cost ratio on unseen data")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Build a belief ROC, back out the implied FN/FP cost ratio, "
                     "and find the best fixed utility for a target cost ratio."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Path to the labeled dataset CSV.")
    p.add_argument("--belief-col", default="belief",
                   help="Column with the elicited probability in [0, 1].")
    p.add_argument("--label-col", default="label",
                   help="Column with the ground-truth binary label.")
    p.add_argument("--decision-col", default=None,
                   help="Optional column with the observed binary decision. "
                        "If omitted, the implied ratio is backed out of the labels.")
    p.add_argument("--cost-ratio", default="1",
                   help="Target cost ratio to evaluate by, as FN/FP. Accepts a "
                        "number (e.g. 5) or an 'FN:FP' string (e.g. 10:1, 1:5).")
    p.add_argument("--output-dir", default=None, type=Path,
                   help="Where to write roc.png and summary.json "
                        "(default: alongside the input file).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip writing the ROC figure.")
    p.add_argument("--title", default="Belief ROC",
                   help="Title for the ROC figure.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cost_ratio = parse_cost_ratio(args.cost_ratio)

    ds = load_dataset(
        args.input,
        belief_col=args.belief_col,
        label_col=args.label_col,
        decision_col=args.decision_col,
    )
    results = analyze(ds, cost_ratio)
    print_report(results)

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    if not args.no_plot:
        bfu = BestFixedUtility(**results["best_fixed_utility"])
        roc_path = out_dir / "roc.png"
        if plot_roc(ds, bfu, results["auroc"], roc_path,
                    implied=results["implied_ratio"], title=args.title):
            print(f"wrote {roc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
