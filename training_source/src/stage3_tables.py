"""Plans B/C/D tables (docs/analiz_plani_v3.md section 3, v4 section 3).

For every completed variant ensemble <arch>__{s3pw,s3xdb,s4multi}:
  family (a): vs the same architecture in Stage 1 (no augmentation)
  family (b): vs the same architecture in Stage 2 (augmentation only)
Endpoint: BUT PDB P_peak F1 (+-150 ms, each lead), subject-level paired bootstrap; Holm within each family.
Secondary columns: pwave P_peak F1, QTDB P_on F1 (for s3xdb: only on the 28 QTDB records not used in training),
LUDB internal-test mean F1 of six on/off points.
Output: results/tables/stage3_*.md/.csv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS  # noqa: E402

F = RESULTS / "final"
T = RESULTS / "tables"
SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
VAR = {"s3pw": "B: aug + P-sensitive loss", "s3xdb": "C: aug + QTDB (non-overlapping)", "s4multi": "D: aug + RDB + Multi-Expert"}
EXCL = re.compile(r"^sel([12]\d\d|8\d\d)$")


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def qtdb_external_only(name, point="P_on"):
    """Pooled F1 of QTDB (each lead) restricted to the 28 records never used for Plan-C training."""
    subj = stats.load_records(name)["qtdb_each_lead"]["subj"][point]
    tp = sum(v["tp"] for k, v in subj.items() if EXCL.match(k))
    fn = sum(v["fn"] for k, v in subj.items() if EXCL.match(k))
    fp = sum(v["fp"] for k, v in subj.items() if EXCL.match(k))
    return 200 * tp / max(2 * tp + fn + fp, 1)


def main():
    rows, fam = [], {"a": [], "b": []}
    for tag, lab in VAR.items():
        for arch in ["cnn_bilstm", "unet_kan", "deq"]:
            name = f"{arch}__{tag}"
            if not (F / f"{name}.json").exists():
                continue
            r = jl(F / f"{name}.json")
            row = {"variant": lab, "architecture": arch, "BUT PDB F1": r["butpdb_each_lead"]["P_peak"]["F1"],
                   "BUT Se/PPV": f"{r['butpdb_each_lead']['P_peak']['Se']:.1f}/{r['butpdb_each_lead']['P_peak']['PPV']:.1f}",
                   "pwave F1": r["pwave_each_lead"]["P_peak"]["F1"],
                   "QTDB P_on F1 (28 ext. records)": qtdb_external_only(name),
                   "LUDB mean6 F1": float(np.mean([r["ludb_test"][q]["F1"] for q in SIX]))}
            rows.append(row)
            for f, ref in [("a", arch), ("b", f"{arch}__s2dissoc")]:
                if (F / f"{ref}_records.pkl").exists():
                    t = stats.compare(name, ref, "butpdb_each_lead", "P_peak")
                    fam[f].append(t)
    df = pd.DataFrame(rows)
    df.round(2).to_csv(T / "stage3_summary.csv", index=False)
    md = ["### Plans B/C/D: summary (each lead, ±150 ms; ensembles of 5 folds)", "", df.round(2).to_markdown(index=False), ""]
    for f, title in [("a", "vs same architecture, Stage 1 (no augmentation)"),
                     ("b", "vs same architecture, Stage 2 (augmentation only)")]:
        if not fam[f]:
            continue
        adj = stats.holm([t["p_boot"] for t in fam[f]])
        md += [f"#### BUT PDB P_peak F1 — family ({f}): {title}; Holm within family", "",
               "| variant | reference | F1 variant | F1 reference | diff [95% CI] | p (bootstrap) | p (Holm) |",
               "|---|---|---|---|---|---|---|"]
        for t, h in zip(fam[f], adj):
            md.append(f"| {t['A']} | {t['B']} | {t['F1_A']:.2f} | {t['F1_B']:.2f} | {t['diff']:+.2f} "
                      f"[{t['diff_CI95'][0]:+.2f}, {t['diff_CI95'][1]:+.2f}] | {t['p_boot']:.4f} | {h:.4f} |")
        md.append("")
    (T / "stage3_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
