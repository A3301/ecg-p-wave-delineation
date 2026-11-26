"""Final model selection and confirmatory tests (plans v3 section 3 and v4 section 3). Written before the
Plan B/C/D results exist.

1. Candidates: every deep ensemble of Stages 1-3 that has results/final/<name>.json
   (Stage 1: 8 architectures; Stage 2: __s2dissoc; Plans B/C/D: __s3pw, __s3xdb, __s4multi).
2. Selection on the development set: highest MIT-BIH P-wave P_peak F1 (each lead, +-150 ms). Ties -> fewer params.
3. Confirmatory tests of the selected configuration vs the Stage-1 best baseline (CNN-BiLSTM, chosen in Stage 1):
     (a) BUT PDB without the 9 pwave-overlapping segments, P_peak F1, each lead, +-150 ms
     (b) Zhejiang, mean of P_on and P_off F1, each lead, +-150 ms  (P-specific endpoint of plan v4)
   Subject-level paired bootstrap (10 000), two-sided; Holm over the two tests.
4. Supporting: the same two tests at the strict tolerance (+-50 ms on BUT PDB, +-40 ms on Zhejiang), reported
   without correction (secondary).
Output: results/tables/final_selection.md/.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS  # noqa: E402

F, X = RESULTS / "final", RESULTS / "final_extra"
BASE = "cnn_bilstm"
B = 10_000


def rec(dirp, name):
    with open(dirp / f"{name}_records.pkl", "rb") as fh:
        return pickle.load(fh)


def boot_diff(sa_list, sb_list, keys, rng):
    """sa_list/sb_list: list of subject dicts (one per point); F1 = mean over points of pooled F1."""
    idx = rng.integers(0, len(keys), (B, len(keys)))

    def f1s(sl):
        point, boot = [], []
        for s in sl:
            t, n, p, _ = stats.subj_arrays(s, keys)
            point.append(stats.f1(t.sum(), n.sum(), p.sum()))
            boot.append(stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1)))
        return float(np.mean(point)), np.mean(boot, 0)
    a, ab = f1s(sa_list)
    b, bb = f1s(sb_list)
    d = ab - bb
    return {"F1_selected": a, "F1_baseline": b, "diff": a - b, "CI95": [float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5))], "p_boot": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "n_subjects": len(keys)}


def main():
    cands = []
    for p in sorted(F.glob("*.json")):
        if p.stem.endswith("_tol50") or p.stem == "wavelet":
            continue
        r = json.loads(p.read_text())
        if "pwave_each_lead" in r:
            cands.append({"model": p.stem, "pwave_F1": r["pwave_each_lead"]["P_peak"]["F1"], "params": r.get("params")})
    cdf = pd.DataFrame(cands).sort_values(["pwave_F1", "params"], ascending=[False, True])
    sel = cdf.iloc[0].model
    out = {"selection_rule": "max pwave P_peak F1 (each lead, +-150 ms)", "selected": sel,
           "candidates": cdf.to_dict("records")}
    rng = np.random.default_rng(0)
    tests = {}
    ra, rb = rec(F, sel), rec(F, BASE)
    sa, sb = ra["butpdb_nooverlap_each_lead"]["subj"]["P_peak"], rb["butpdb_nooverlap_each_lead"]["subj"]["P_peak"]
    keys = sorted(set(sa) | set(sb))
    tests["BUT PDB no-overlap P_peak (+-150 ms)"] = boot_diff([sa], [sb], keys, rng)
    zj_ok = (X / f"{sel}_records.pkl").exists() and (X / f"{BASE}_records.pkl").exists()
    if zj_ok:
        xa, xb = rec(X, sel), rec(X, BASE)
        for key, lab in [("zj_each_lead", "Zhejiang mean(P_on,P_off) F1 (+-150 ms)"),
                         ("zj_each_lead_tol40", "Zhejiang mean(P_on,P_off) F1 (+-40 ms) [secondary]")]:
            if key in xa and key in xb:
                la = [xa[key]["subj"][q] for q in ("P_on", "P_off")]
                lb = [xb[key]["subj"][q] for q in ("P_on", "P_off")]
                keys = sorted(set(la[0]) | set(lb[0]))
                tests[lab] = boot_diff(la, lb, keys, rng)
    t50a, t50b = F / f"{sel}_tol50_records.pkl", F / f"{BASE}_tol50_records.pkl"
    if t50a.exists() and t50b.exists():
        sa = pickle.load(open(t50a, "rb"))["butpdb_nooverlap_each_lead"]["subj"]["P_peak"]
        sb = pickle.load(open(t50b, "rb"))["butpdb_nooverlap_each_lead"]["subj"]["P_peak"]
        tests["BUT PDB no-overlap P_peak (+-50 ms) [secondary]"] = boot_diff([sa], [sb], sorted(set(sa) | set(sb)), rng)
    conf = [k for k in tests if "[secondary]" not in k]
    adj = stats.holm([tests[k]["p_boot"] for k in conf])
    for k, a in zip(conf, adj):
        tests[k]["p_holm"] = a
    out["tests_vs_" + BASE] = tests
    (RESULTS / "tables" / "final_selection.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    md = [f"### Final selection (development set: MIT-BIH P-wave) and confirmatory tests", "",
          f"Selected configuration: **{sel}** (rule: {out['selection_rule']}).", "",
          cdf.head(12).to_markdown(index=False), "", f"Confirmatory tests vs {BASE} (Stage 1 best baseline):", "",
          "| test | F1 selected | F1 baseline | diff [95% CI] | p (bootstrap) | p (Holm) | n subjects |",
          "|---|---|---|---|---|---|---|"]
    for k, t in tests.items():
        md.append(f"| {k} | {t['F1_selected']:.2f} | {t['F1_baseline']:.2f} | {t['diff']:+.2f} "
                  f"[{t['CI95'][0]:+.2f}, {t['CI95'][1]:+.2f}] | {t['p_boot']:.4f} | "
                  f"{t.get('p_holm', float('nan')):.4f} | {t['n_subjects']} |")
    (RESULTS / "tables" / "final_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
