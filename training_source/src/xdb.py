"""Plan C: cross-database training data from QTDB (q1c), excluding every record that shares a source database
with the external P-wave test sets (BUT PDB, MIT-BIH P-wave annotations):
  excluded: sel1xx/sel2xx (MIT-BIH Arrhythmia DB, 15 records) and sel8xx (MIT-BIH Supraventricular Arrhythmia DB,
  13 records). The 77 remaining records (European ST-T, MIT-BIH ST change, NSR, long-term ST, sudden death) are used
  for TRAINING ONLY; QTDB evaluation of Plan-C models is restricted to the 28 excluded records.

Labels (QTDB annotates only selected consecutive beats):
  - every sample is 'ignore' (-1) except inside annotated-beat windows [QRS peak - 350 ms, QRS peak + 550 ms]
  - inside the windows: background 0, P on..off = 1, QRS on..off = 2, T on..off = 3 when T onset is annotated;
    when T onset is missing, the segment QRS off..T off is set to ignore (its T boundary is unknown).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

EXCLUDE = re.compile(r"^sel([12]\d\d|8\d\d)$")


def is_excluded(record: str) -> bool:
    return bool(EXCLUDE.match(record))


def _pairs(on, off, max_len):
    """Pair each offset with the latest onset before it within max_len samples."""
    out = []
    on = np.sort(on)
    for e in np.sort(off):
        cand = on[(on < e) & (on >= e - max_len)]
        out.append((int(cand[-1]) if len(cand) else None, int(e)))
    return out


def qtdb_train_samples():
    fs = data.FS
    out = []
    for s in data.get("qtdb"):
        if is_excluded(s.record):
            continue
        n = len(s.x)
        y = np.full(n, -1, dtype=np.int8)
        pts = s.points
        for q in pts["QRS_peak"]:
            y[max(0, q - int(0.35 * fs)):min(n, q + int(0.55 * fs))] = 0
        for on, off in _pairs(pts["P_on"], pts["P_off"], int(0.25 * fs)):
            if on is not None:
                y[on:off + 1] = 1
        qrs = _pairs(pts["QRS_on"], pts["QRS_off"], int(0.25 * fs))
        for on, off in qrs:
            if on is not None:
                y[on:off + 1] = 2
        qrs_off = np.array([e for _, e in qrs], int)
        for on, off in _pairs(pts["T_on"], pts["T_off"], int(0.45 * fs)):
            if on is not None:
                y[on:off + 1] = 3
            else:
                prev = qrs_off[qrs_off < off]
                if len(prev):
                    y[prev[-1] + 1:off + 1] = -1
        lab = np.flatnonzero(y >= 0)
        if not len(lab):
            continue
        s2 = data.LeadSample("QTDB", s.record, f"QTDB:{s.record}", s.lead, s.x, y, s.points,
                             (int(lab.min()), int(lab.max()) + 1), {"source": "qtdb_train"})
        out.append(s2)
    return out


if __name__ == "__main__":
    smp = qtdb_train_samples()
    recs = sorted({s.record for s in smp})
    lab = np.concatenate([s.y[s.span[0]:s.span[1]] for s in smp])
    print(len(recs), "records,", len(smp), "lead-samples; label share in span:",
          {c: round(float((lab == c).mean()), 3) for c in (-1, 0, 1, 2, 3)})
