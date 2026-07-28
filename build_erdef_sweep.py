"""Build the new-definition ("ER required") sweep used for the rebuilt figures.

The reviewer noted that coding every C/D boundary case as *requiring* emergency
care (positive) while only counting a "D" recommendation as a hit turns a
clinically-acceptable "C" recommendation into a false negative. C/D only
establishes that emergency care is *acceptable*, not *required*.

This script produces a copy of the factorial sweep restricted to a well-defined
binary emergency endpoint:

  * Sample : the 960 original-vignette variants (the 18 supplementary "textbook"
             emergency vignettes are excluded so the endpoint matches the
             original Nature study's 960-case dataset).
  * Positive (requires ER) : gold_standard == "D"
  * Negative               : gold_standard in {A, B, C, A/B, B/C}
  * Excluded (ambiguous)   : gold_standard == "C/D"

For every kept row we recompute needs_er_gold = 1 if gold_standard == "D" else 0.
fit.json / settings.json are copied unchanged: the recovered cost ratios they
hold are NOT a function of the gold label, so they must stay identical to the
main-text (full-data) figures for cross-figure consistency.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

SRC = Path("revealed_preferences/2026-07-01_factorial")
DST = Path("revealed_preferences/erdef_factorial_960")

# 18 supplementary vignettes added to the expanded set (textbook emergencies).
SUPPLEMENTARY = {
    "E28", "E29", "E30", "E31",
    "F28", "F29", "F30", "F31",
    "MH4", "MH5", "MH6", "MH7", "MH8",
    "NH4", "NH5", "NH6", "NH7", "NH8",
}


def keep_row(row: dict) -> bool:
    if row["vignette_id"] in SUPPLEMENTARY:
        return False
    if row["gold_standard"] == "C/D":
        return False
    return True


def filter_results(src_csv: Path, dst_csv: Path) -> tuple[int, int]:
    with open(src_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows_in = list(reader)

    rows_out = []
    for r in rows_in:
        if not keep_row(r):
            continue
        r = dict(r)
        r["needs_er_gold"] = "1" if r["gold_standard"] == "D" else "0"
        rows_out.append(r)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_in), len(rows_out)


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    cfg_dirs = sorted(p.parent for p in SRC.glob("*/results.csv"))
    print(f"Building {DST} from {len(cfg_dirs)} configs\n")
    for cfg in cfg_dirs:
        out_cfg = DST / cfg.name
        out_cfg.mkdir(parents=True, exist_ok=True)
        n_in, n_out = filter_results(cfg / "results.csv", out_cfg / "results.csv")
        # copy the untouched companion files (recovered cost ratios etc.)
        for extra in ("fit.json", "settings.json"):
            src = cfg / extra
            if src.exists():
                shutil.copy2(src, out_cfg / extra)
        print(f"  {cfg.name:32s} rows {n_in:6d} -> {n_out:6d}")
    print(f"\nDone. Wrote filtered sweep to {DST}")


if __name__ == "__main__":
    main()
