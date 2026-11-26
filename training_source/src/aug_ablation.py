"""Component analysis of the augmentation: which part of it does the work?

U-Net-KAN, seed 1, five-fold ensembles: Stage 1 (no augmentation), both (insert + delete, Stage 2), insert only,
delete only, and the QRS-coupled sham (same templates and maximum number of pastes as insert, centred on existing P waves).
Endpoints: BUT PDB P-peak F1 (each lead; all segments and without overlap), MIT-BIH P-wave P-peak F1, QTDB P-onset F1
(each lead), LUDB mean F1 of six points, Zhejiang P-onset F1 (each lead); sensitivity for the blocked P waves of record
231 (BUT PDB and MIT-BIH P-wave annotation) and for QRS-coupled P waves on BUT PDB.
Differences: paired subject-level bootstrap (10 000; fresh rng(0) per comparison), against Stage 1 and against insert.
Models whose evaluation is missing are skipped.
Output: results/tables/aug_ablation.md, results/aug_ablation.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
MODELS = [("Stage 1", "unet_kan"), ("insert + delete (Stage 2)", "unet_kan__s2dissoc"), ("insert only", "unet_kan__s5ins"),
          ("delete only", "unet_kan__s5del"), ("QRS-coupled sham", "unet_kan__s5sham")]
FINAL_EP = [("butpdb_each_lead", "P_peak", "BUT PDB"), ("butpdb_nooverlap_each_lead", "P_peak", "BUT PDB, no overlap"),
            ("pwave_each_lead", "P_peak", "MIT-BIH P-wave"), ("qtdb_each_lead", "P_on", "QTDB P onset")]


def subj(path, part, q):
    with open(path, "rb") as fh:
        return pickle.load(fh)[part]["subj"][q]


def pooled(sb):
    keys = sorted(sb)
    tp, fn, fp, _ = stats.subj_arrays(sb, keys)
    return float(stats.f1(tp.sum(), fn.sum(), fp.sum()))


def diff(sa, sb):
    keys = sorted(set(sa) | set(sb))
    ta, na, pa, _ = stats.subj_arrays(sa, keys)
    tb, nb, pb, _ = stats.subj_arrays(sb, keys)
    idx = np.random.default_rng(0).integers(0, len(keys), (stats.B, len(keys)))
    d = stats.f1(ta[idx].sum(1), na[idx].sum(1), pa[idx].sum(1)) - stats.f1(tb[idx].sum(1), nb[idx].sum(1), pb[idx].sum(1))
    return float(stats.f1(ta.sum(), na.sum(), pa.sum()) - stats.f1(tb.sum(), nb.sum(), pb.sum())), np.nanpercentile(d, [2.5, 97.5]).tolist()


def mean6(m):
    j = json.loads((R / "final" / f"{m}.json").read_text())
    return float(np.mean([j["ludb_test"][q]["F1"] for q in SIX]))


def mean6_diff(a, b):
    """Difference of the mean F1 over six points, same subject resample for all points (as make_tables.h1_h2)."""
    ra, rb = stats.load_records(a)["ludb_test"]["subj"], stats.load_records(b)["ludb_test"]["subj"]
    keys = sorted(ra["P_on"])
    idx = np.random.default_rng(0).integers(0, len(keys), (stats.B, len(keys)))
    fa, fb, da, db = [], [], [], []
    for q in SIX:
        ta, na, pa, _ = stats.subj_arrays(ra[q], keys)
        tb, nb, pb, _ = stats.subj_arrays(rb[q], keys)
        fa.append(stats.f1(ta.sum(), na.sum(), pa.sum()))
        fb.append(stats.f1(tb.sum(), nb.sum(), pb.sum()))
        da.append(stats.f1(ta[idx].sum(1), na[idx].sum(1), pa[idx].sum(1)))
        db.append(stats.f1(tb[idx].sum(1), nb[idx].sum(1), pb[idx].sum(1)))
    d = np.mean(da, 0) - np.mean(db, 0)
    return float(np.mean(fa) - np.mean(fb)), np.percentile(d, [2.5, 97.5]).tolist()


def p_flags(m, db):
    f = R / ("mechanism_detail" if db == "butpdb" else "mechanism_detail_pwave") / f"{m}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        rows = pickle.load(fh)
    out = {}
    for name, sel, kind in [("231 blocked", lambda r: r["subject"].endswith(":231"), "dis"), ("coupled", lambda r: True, "cpl")]:
        found = []
        for r in rows:
            if not sel(r):
                continue
            d = ~np.asarray(r["coupled"], bool)
            k = d if kind == "dis" else ~d
            found.append(np.asarray(r["found"], bool)[k])
        fo = np.concatenate(found) if found else np.zeros(0, bool)
        out[name] = (100 * fo.mean() if len(fo) else float("nan"), len(fo))
    return out


def main():
    avail = [(lab, m) for lab, m in MODELS if (R / "final" / f"{m}_records.pkl").exists()]
    res = {"models": {}, "diffs": []}
    head = ["model"] + [e[2] for e in FINAL_EP] + ["LUDB mean of 6", "Zhejiang P onset", "231 blocked P, BUT PDB",
                                                   "231 blocked P, P-wave set", "QRS-coupled P, BUT PDB"]
    md = ["### Augmentation components (U-Net-KAN, seed 1; F1 % each lead, ±150 ms; sensitivity %)", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for lab, m in avail:
        row = {}
        for part, q, nm in FINAL_EP:
            row[nm] = pooled(subj(R / "final" / f"{m}_records.pkl", part, q))
        row["LUDB mean of 6"] = mean6(m)
        fz = R / "final_extra" / f"{m}_records.pkl"
        row["Zhejiang P onset"] = pooled(subj(fz, "zj_each_lead", "P_on")) if fz.exists() else float("nan")
        fb, fp = p_flags(m, "butpdb"), p_flags(m, "pwave")
        row["231 blocked P, BUT PDB"] = fb["231 blocked"][0] if fb else float("nan")
        row["231 blocked P, P-wave set"] = fp["231 blocked"][0] if fp else float("nan")
        row["QRS-coupled P, BUT PDB"] = fb["coupled"][0] if fb else float("nan")
        res["models"][lab] = row
        md.append(f"| {lab} | " + " | ".join("–" if np.isnan(row[h]) else f"{row[h]:.2f}" for h in head[1:]) + " |")
    md += ["", "| comparison | " + " | ".join(e[2] + ": Δ [95% CI]" for e in FINAL_EP) + " | Zhejiang P onset: Δ [95% CI] | LUDB mean of 6: Δ [95% CI] |",
           "|---|" + "---|" * (len(FINAL_EP) + 2)]
    names = dict(avail)
    pairs = [(a, "Stage 1") for a, _ in avail if a != "Stage 1"] + \
            [(a, "insert only") for a in ("insert + delete (Stage 2)", "delete only", "QRS-coupled sham") if a in names and "insert only" in names]
    for a, b in pairs:
        cells, rec = [], {"A": a, "B": b}
        for part, q, nm in FINAL_EP:
            d, ci = diff(subj(R / "final" / f"{names[a]}_records.pkl", part, q), subj(R / "final" / f"{names[b]}_records.pkl", part, q))
            cells.append(f"{d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
            rec[nm] = [d, ci]
        fa, fb = R / "final_extra" / f"{names[a]}_records.pkl", R / "final_extra" / f"{names[b]}_records.pkl"
        if fa.exists() and fb.exists():
            d, ci = diff(subj(fa, "zj_each_lead", "P_on"), subj(fb, "zj_each_lead", "P_on"))
            cells.append(f"{d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
            rec["Zhejiang P onset"] = [d, ci]
        else:
            cells.append("–")
        d6, ci6 = mean6_diff(names[a], names[b])
        cells.append(f"{d6:+.2f} [{ci6[0]:+.2f}, {ci6[1]:+.2f}]")
        rec["LUDB mean of 6"] = [d6, ci6]
        res["diffs"].append(rec)
        md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    (R / "aug_ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (R / "tables" / "aug_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written", [lab for lab, _ in avail])


if __name__ == "__main__":
    main()
