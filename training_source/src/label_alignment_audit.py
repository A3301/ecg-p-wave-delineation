"""Label-signal alignment audit for every database (catches shifted / wrong-version labels).

Per lead-signal: ratio = mean |dx/dt| inside QRS-labelled samples / mean |dx/dt| in labelled background.
Well-aligned QRS labels give large ratios (QRS is the steepest wave); shifted labels give ~1.
Reference distribution: LUDB. A record is flagged when its median ratio over leads < 1.5.
Output: results/label_alignment_audit.json, printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import extra_data  # noqa: E402
from paths import RESULTS  # noqa: E402

THR = 1.5


def ratio(s):
    d = np.abs(np.diff(s.x, prepend=s.x[0]))
    y = s.y
    q, b = y == 2, y == 0
    if q.sum() < 5 or b.sum() < 20:
        return np.nan
    return float(d[q].mean() / (d[b].mean() + 1e-9))


def audit(name, samples):
    per_rec = {}
    for s in samples:
        r = ratio(s)
        if not np.isnan(r):
            per_rec.setdefault(s.record, []).append(r)
    med = {k: float(np.median(v)) for k, v in per_rec.items()}
    vals = np.array(list(med.values()))
    flagged = sorted([k for k, v in med.items() if v < THR], key=lambda k: med[k])
    out = {"n_records": len(med), "median": float(np.median(vals)), "p5": float(np.percentile(vals, 5)),
           "min": float(vals.min()), "n_flagged": len(flagged), "flagged": {k: med[k] for k in flagged}}
    print(f"{name:5s} records {len(med):5d} | QRS/bg slope ratio median {out['median']:.2f} p5 {out['p5']:.2f} "
          f"min {out['min']:.2f} | flagged(<{THR}) {len(flagged)}", flush=True)
    return out


def main():
    res = {"threshold": THR, "LUDB": audit("LUDB", data.get("ludb"))}
    for nm in ["rdb", "me", "isp", "zj"]:
        res[nm] = audit(nm, extra_data.get(nm))
    (RESULTS / "label_alignment_audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
