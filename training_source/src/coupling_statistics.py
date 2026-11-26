"""Additional analysis (plan v5, 3.1): subject-aware statistics for QRS-dissociated P-wave detection on BUT PDB.

Uses the P-wave-level flags of src/coupling_detail.py. For each augmented/variant model vs the Stage-1 model of the
same architecture:
  - dissociated-P sensitivity for: all leads; first lead only; record 231 only; all subjects except 231;
    excluding reference P waves with another reference P within 100 ms (possible duplicate labels);
  - exact paired McNemar test on the same P waves (discordant pairs counted directly);
  - subject-level sign test (subjects with at least one dissociated P, improved vs worsened);
  - subject-cluster bootstrap 95% CI of the sensitivity difference (10 000 resamples).
Output: results/tables/coupling_statistics.md
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS  # noqa: E402

D = RESULTS / "mechanism_detail"
DB = "butpdb"
PAIRS = [(f"{a}__{t}", a) for t in ["s2dissoc"] for a in ["unet", "cnn_bilstm", "unet_kan", "deq"]] + \
        [(f"{a}__{t}", a) for t in ["s3xdb", "s4multi"] for a in ["cnn_bilstm", "unet_kan", "deq"]] +         [(f"{a}__s5cd", a) for a in ["cnn_bilstm", "unet_kan"]] +         [(f"unet_kan__{t}", "unet_kan") for t in ["s5ins", "s5del", "s5sham"]]   # additional analyses (plan v5)


def load(n):
    with open(D / f"{n}.pkl", "rb") as fh:
        return pickle.load(fh)


def flat(rows, sel):
    """Concatenate per-P arrays over rows passing sel(row); returns subject array and dicts of flags."""
    subj, found, dis, dup = [], [], [], []
    for r in rows:
        if not sel(r):
            continue
        n = len(r["found"])
        subj += [r["subject"]] * n
        found.append(r["found"]); dis.append(~r["coupled"]); dup.append(r["near_dup"])
    if not found:
        return np.array([]), np.array([], bool), np.array([], bool), np.array([], bool)
    return np.array(subj), np.concatenate(found), np.concatenate(dis), np.concatenate(dup)


def mcnemar(a, b):
    """a, b: boolean detection arrays on the same P waves; exact two-sided McNemar."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    m = n10 + n01
    return n10, n01, (1.0 if m == 0 else float(min(1, 2 * binom.cdf(min(n10, n01), m, 0.5))))


def analyse(var, base, sel, mask_fn, rng):
    rv, rb = load(var), load(base)
    s, fv, d, dup = flat(rv, sel)
    _, fb, _, _ = flat(rb, sel)
    k = mask_fn(d, dup)
    s, fv, fb = s[k], fv[k], fb[k]
    if len(s) == 0:
        return None
    n10, n01, p = mcnemar(fv, fb)
    subs = np.unique(s)
    per = [(fv[s == u].mean(), fb[s == u].mean()) for u in subs]
    up = sum(a > b for a, b in per); down = sum(a < b for a, b in per)
    sign_p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    idx = rng.integers(0, len(subs), (10000, len(subs)))
    tv = np.array([fv[s == u].sum() for u in subs]); tb = np.array([fb[s == u].sum() for u in subs])
    nn = np.array([(s == u).sum() for u in subs])
    diff = 100 * (tv[idx].sum(1) / nn[idx].sum(1) - tb[idx].sum(1) / nn[idx].sum(1))
    return {"n": len(s), "subjects": len(subs), "se_var": 100 * fv.mean(), "se_base": 100 * fb.mean(),
            "n10": n10, "n01": n01, "p_mcnemar": p, "up": up, "down": down, "p_sign": sign_p,
            "ci": (np.percentile(diff, 2.5), np.percentile(diff, 97.5))}


def main():
    global D, DB
    if len(sys.argv) > 1 and sys.argv[1] == "pwave":
        D, DB = RESULTS / "mechanism_detail_pwave", "pwave"
    rng = np.random.default_rng(0)
    md = [f"### Subject-aware statistics for QRS-dissociated P waves ({'MIT-BIH P-wave' if DB == 'pwave' else 'BUT PDB'}, ±150 ms)", "",
          "Se: sensitivity for dissociated P waves, variant vs Stage 1 (same architecture). McNemar: exact, on the same P "
          "waves (n10 = found only by the variant, n01 = only by Stage 1). Sign test over subjects with ≥ 1 dissociated P. "
          "CI: subject-cluster bootstrap of the Se difference.", ""]
    subsets = [("all leads, all subjects", lambda r: True, lambda d, u: d),
               ("first lead only", lambda r: r["lead"] in ("ch0", "MLII"), lambda d, u: d),
               ("record 231 only", lambda r: r["subject"].endswith(":231"), lambda d, u: d),
               ("all subjects except 231", lambda r: not r["subject"].endswith(":231"), lambda d, u: d),
               ("near-duplicate labels removed", lambda r: True, lambda d, u: d & ~u)]
    if DB == "pwave":
        subsets.insert(3, ("record 119 only", lambda r: r["subject"].endswith(":119"), lambda d, u: d))
    for name, sel, mk in subsets:
        md += [f"#### {name}", "", "| variant | n P (subjects) | Se Stage 1 → variant | n10 / n01 | McNemar p | subjects up/down (sign p) | Se diff 95% CI (cluster) |",
               "|---|---|---|---|---|---|---|"]
        for var, base in PAIRS:
            if not ((D / f"{var}.pkl").exists() and (D / f"{base}.pkl").exists()):
                continue
            r = analyse(var, base, sel, mk, np.random.default_rng([ord(c) for c in var + name]))  # order-independent
            if r is None:
                continue
            ci = "– (one subject)" if r["subjects"] < 2 else f"{r['ci'][0]:+.1f} to {r['ci'][1]:+.1f}"
            sign = "–" if r["subjects"] < 2 else f"{r['up']}/{r['down']} ({r['p_sign']:.2g})"
            md.append(f"| {var} | {r['n']} ({r['subjects']}) | {r['se_base']:.1f} → {r['se_var']:.1f} | {r['n10']} / {r['n01']} | "
                      f"{r['p_mcnemar']:.2g} | {sign} | {ci} |")
        md.append("")
    # per-subject table (Stage 1 vs augmentation, all four architectures pooled per subject are NOT pooled: one row per subject x arch)
    archs = [a for a in ["unet", "cnn_bilstm", "unet_kan", "deq"] if (D / f"{a}.pkl").exists() and (D / f"{a}__s2dissoc.pkl").exists()]
    md += ["#### Per subject: dissociated P waves (lead-level count) and sensitivity, Stage 1 → augmentation", "",
           "| subject | n dissociated P | " + " | ".join(archs) + " |", "|---|---|" + "---|" * len(archs)]
    subs = None
    for a in archs:
        sb, fb, d, _ = flat(load(a), lambda r: True); _, fv, _, _ = flat(load(a + "__s2dissoc"), lambda r: True)
        if subs is None:
            subs = {u: [int((sb[d] == u).sum())] for u in np.unique(sb[d])}
        for u in subs:
            k = d & (sb == u)
            subs[u].append(f"{100 * fb[k].mean():.0f} → {100 * fv[k].mean():.0f}")
    for u, v in sorted((subs or {}).items(), key=lambda kv: -kv[1][0]):
        md.append(f"| {u.split(':')[-1]} | {v[0]} | " + " | ".join(v[1:]) + " |")
    (RESULTS / "tables" / f"coupling_statistics{'_pwave' if DB == 'pwave' else ''}.md").write_text("\n".join(md), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
