"""Rebuild the gold-label-dependent paper figures under the new ER endpoint.

Run build_erdef_sweep.py first to create the filtered 960-variant / pure-D sweep.

The new emergency endpoint (see build_erdef_sweep.py) excludes the C/D boundary
cases (where emergency care is *acceptable* but not *required*) and the 18
supplementary textbook vignettes, leaving 576 variants with an unambiguous
"D == requires ER" label. Only the figures whose content depends on the gold
label are regenerated here:

    fig1_emergency_triage_performance.png
    fig3_roc_grid_{gpt,deepseek,claude}.png
    fig_calibration_curves.png
    fig_belief_distributions.png   (yellow "best fixed threshold" marker uses gold)

Everything is written to nature_medicine_paper/new_figures/. The gold-independent
figures (recovered utilities/thresholds, consistency, prompt schematics) are
carried over unchanged by copy_independent_figures.py.
"""
from __future__ import annotations

from pathlib import Path

import analyze_sweep as A

# Importing this module installs the full seven-utility configuration on the
# shared analysis modules (A/rp), exactly as the main-text build does.
import make_full_utility_figures as F
import make_calibration_figure as CAL
import make_threshold_figure as TH

FILTERED = Path("revealed_preferences/erdef_factorial_960")
NEW_OUT = Path("nature_medicine_paper/new_figures")


def main() -> None:
    NEW_OUT.mkdir(parents=True, exist_ok=True)

    # Redirect every generator to the filtered sweep and the new output folder.
    F.SWEEP = FILTERED
    F.BELIEF_SWEEP = FILTERED
    F.OUT = NEW_OUT
    CAL.BELIEF_SWEEP = FILTERED
    CAL.OUT = NEW_OUT
    TH.BELIEF_SWEEP = FILTERED
    TH.OUT = NEW_OUT

    by_name = {
        d.parent.name: A.analyse_config(d.parent)
        for d in sorted(FILTERED.glob("*/results.csv"))
    }

    print("== Figure 1 (deployed-tool ROC) ==")
    F.figure1()

    print("\n== Figure 3 (belief-oracle ROC grids) ==")
    F.figure3(by_name, F.CONFIG_ORDER, NEW_OUT)

    print("\n== Calibration curves ==")
    CAL.build()

    print("\n== Belief distributions ==")
    TH.build_belief_grid()

    print(f"\nRebuilt gold-dependent figures written to {NEW_OUT}")


if __name__ == "__main__":
    main()
