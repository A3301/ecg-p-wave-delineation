"""P–QRS dissociation augmentation (Stage 2 method).

Motivation (Stage 1 failure analysis): models trained on LUDB, where almost every P wave sits ~120-200 ms before a
QRS, learn "P = the wave right before the QRS" and miss P waves that are dissociated from the QRS (AV block),
ectopic, or hidden in T waves on external databases.

Operation on one training crop (single lead, labels y: 0 bg, 1 P, 2 QRS, 3 T, -1 ignore), using real P-wave
templates taken from the SAME lead of the SAME subject (so amplitude/morphology stay realistic and no information
from other subjects or external databases is used):
  INSERT : paste 1..n_max extra P templates at random positions that do not touch a QRS or an ignore region; they may
           overlap T waves or ST/TP segments (simulates dissociated P, P-on-T, blocked P). The template is baseline-
           corrected (chord subtracted) and ADDED to the signal; its support is labelled P (P has priority over T).
  REMOVE : with probability p_remove, an annotated P wave in the crop is replaced by the straight line joining its
           onset and offset values (+ small noise) and relabelled background (simulates absent P: junctional
           rhythm, AF-like beats).
  Amplitude of inserted templates is scaled by U(0.7, 1.3).
All parameters are fixed a priori (docs/analiz_plani_v2.md); none is tuned on external data.
"""
from __future__ import annotations

import numpy as np

QRS_GUARD = 10  # samples (40 ms at 250 Hz) kept free around any QRS label


def p_templates(x: np.ndarray, y: np.ndarray, min_len=8, max_len=50):
    """Baseline-corrected P-wave templates from a labelled lead (runs of y == 1 fully annotated)."""
    out = []
    m = np.concatenate([[False], y == 1, [False]])
    d = np.diff(m.astype(np.int8))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if min_len <= e - s <= max_len:
            seg = x[s:e].astype(np.float64)
            out.append(seg - np.linspace(seg[0], seg[-1], len(seg)))
    return out


def augment(xc: np.ndarray, yc: np.ndarray, templates, rng, n_max=3, p_remove=0.3):
    """xc, yc: crop (copied). Returns augmented (x, y)."""
    x = xc.astype(np.float64).copy()
    y = yc.copy()
    n = len(x)
    # REMOVE some existing P waves
    m = np.concatenate([[False], y == 1, [False]])
    d = np.diff(m.astype(np.int8))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if s > 0 and e < n and rng.random() < p_remove:
            x[s:e] = np.linspace(x[s - 1], x[e], e - s) + rng.normal(0, 0.02, e - s)
            y[s:e] = 0
    # INSERT dissociated P waves
    if templates and n_max > 0:
        blocked = (y == 2) | (y < 0)
        if blocked.any():  # dilate QRS / ignore by the guard band
            k = np.convolve(blocked.astype(float), np.ones(2 * QRS_GUARD + 1), mode="same") > 0
        else:
            k = blocked
        for _ in range(int(rng.integers(1, n_max + 1))):
            t = templates[int(rng.integers(len(templates)))] * rng.uniform(0.7, 1.3)
            L = len(t)
            if L >= n:
                continue
            for _try in range(10):
                pos = int(rng.integers(0, n - L))
                if not k[pos:pos + L].any() and not (y[pos:pos + L] == 1).any():
                    x[pos:pos + L] += t
                    y[pos:pos + L] = 1
                    k[max(0, pos - QRS_GUARD):pos + L + QRS_GUARD] = True  # no overlap between inserted P waves
                    break
    return x.astype(np.float32), y


def augment_sham(xc: np.ndarray, yc: np.ndarray, templates, rng, n_max=3):
    """QRS-coupled sham control (revision analysis A-2). 1..n_max annotated P waves of the crop are replaced, centred
    at their own position, by a random template of the same lead (amplitude U(0.7, 1.3)): the same templates and the
    same number of pastes as INSERT, but every P wave keeps its coupling to the following QRS. No P wave is removed."""
    x = xc.astype(np.float64).copy()
    y = yc.copy()
    n = len(x)
    if not templates or n_max <= 0:
        return x.astype(np.float32), y
    m = np.concatenate([[False], y == 1, [False]])
    d = np.diff(m.astype(np.int8))
    runs = [(s, e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if s > 0 and e < n]
    if not runs:
        return x.astype(np.float32), y
    blocked = (y == 2) | (y < 0)
    k = np.convolve(blocked.astype(float), np.ones(2 * QRS_GUARD + 1), mode="same") > 0 if blocked.any() else blocked
    for j in rng.permutation(len(runs))[:int(rng.integers(1, n_max + 1))]:
        s, e = runs[j]
        t = templates[int(rng.integers(len(templates)))] * rng.uniform(0.7, 1.3)
        L = len(t)
        ns = (s + e) // 2 - L // 2
        if ns < 1 or ns + L >= n or k[ns:ns + L].any():
            continue
        x[s:e] = np.linspace(x[s - 1], x[e], e - s)
        y[s:e] = 0
        x[ns:ns + L] += t
        y[ns:ns + L] = 1
    return x.astype(np.float32), y
