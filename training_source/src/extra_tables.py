"""Tables for the new external sets (plan v4 section 2): Zhejiang (independent; PVC/VT) and ISP (same institution as
Multi-Expert, disjoint records). All ensembles with results/final_extra/<name>_records.pkl.

- per model: F1 of P_on, P_off, mean of 6 on/off points, with subject-level bootstrap 95% CI (each lead, +-150 ms);
  Zhejiang also at +-40 ms (Lim et al. 2026 tolerance)
- per architecture: variant vs Stage-1 model on Zhejiang mean(P_on,P_off) F1 (exploratory; Holm within family)
Output: results/tables/extra_sets.md/.csv, results/tables/extra_sets_variants.md
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS  # noqa: E402

X = RESULTS / "final_extra"
SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
B = 10_000


def load(name):
    with open(X / f"{name}_records.pkl", "rb") as fh:
        return pickle.load(fh)


def f1_ci(subjs, rng):
    keys = sorted(set().union(*[set(s) for s in subjs]))
    idx = rng.integers(0, len(keys), (B, len(keys)))
    pt, bt = [], []
    for s in subjs:
        t, n, p, _ = stats.subj_arrays(s, keys)
        pt.append(stats.f1(t.sum(), n.sum(), p.sum()))
        bt.append(stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1)))
    b = np.mean(bt, 0)
    return float(np.mean(pt)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rng = np.random.default_rng(0)
    names = sorted(p.stem.replace("_records", "") for p in X.glob("*_records.pkl"))
    rows = []
    for nm in names:
        r = load(nm)
        for key, lab in [("zj_each_lead", "Zhejiang ±150 ms"), ("zj_each_lead_tol40", "Zhejiang ±40 ms"),
                         ("isp_each_lead", "ISP ±150 ms"), ("zj_lead_ii", "Zhejiang lead II ±150 ms"),
                         ("isp_lead_ii", "ISP lead II ±150 ms")]:
            if key not in r:
                continue
            s = r[key]["subj"]
            row = {"model": nm, "set": lab}
            rng = np.random.default_rng([ord(c) for c in nm + key])  # fixed per model and set
            for q in ["P_on", "P_off"]:
                v, lo, hi = f1_ci([s[q]], rng)
                row[f"{q} F1 [95% CI]"] = f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
            v, lo, hi = f1_ci([s[q] for q in SIX if q in s], rng)
            row["mean of 6 F1 [95% CI]"] = f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tables" / "extra_sets.csv", index=False)
    md = ["### New external sets: Zhejiang (independent, PVC/VT) and ISP (same institution as Multi-Expert, disjoint "
          "records); each lead unless stated; subject-level bootstrap 95% CI", ""]
    for st, g in df.groupby("set", sort=False):
        md += [f"#### {st}", "", g.drop(columns=["set"]).to_markdown(index=False), ""]
    (RESULTS / "tables" / "extra_sets.md").write_text("\n".join(md), encoding="utf-8")
    # variants vs Stage 1 within architecture on Zhejiang mean(P_on, P_off)
    vrows = []
    for nm in names:
        if "__" not in nm or nm == "wavelet":
            continue
        base = nm.split("__")[0]
        if base not in names:
            continue
        a, b = load(nm)["zj_each_lead"]["subj"], load(base)["zj_each_lead"]["subj"]
        keys = sorted(set(a["P_on"]) | set(b["P_on"]))
        idx = np.random.default_rng([ord(c) for c in nm]).integers(0, len(keys), (B, len(keys)))

        def fb(s):
            pt, bt = [], []
            for q in ("P_on", "P_off"):
                t, n, p, _ = stats.subj_arrays(s[q], keys)
                pt.append(stats.f1(t.sum(), n.sum(), p.sum()))
                bt.append(stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1)))
            return float(np.mean(pt)), np.mean(bt, 0)
        fa, ba = fb(a)
        fb_, bb = fb(b)
        d = ba - bb
        vrows.append({"variant": nm, "vs": base, "F1 variant": fa, "F1 base": fb_, "diff": fa - fb_,
                      "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                      "p": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    if vrows:
        v = pd.DataFrame(vrows)
        v["p_holm"] = stats.holm(v.p.tolist())
        v["diff [95% CI]"] = v.apply(lambda r: f"{r['diff']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1)
        out = v[["variant", "vs", "F1 variant", "F1 base", "diff [95% CI]", "p", "p_holm"]].round(4)
        (RESULTS / "tables" / "extra_sets_variants.md").write_text(
            "### Zhejiang mean(P_on, P_off) F1: variant vs same-architecture Stage-1 model (exploratory, Holm)\n\n"
            + out.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_string()[:3000])


if __name__ == "__main__":
    main()
