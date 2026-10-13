#!/usr/bin/env python3
"""
Supplementary script
TOPSIS-RAD: Ranking According to Desires -- Journal of Multi-criteria Decision Analysis 

Reproduces every numerical result reported in the paper:
  * Examples A, B, C (Tables 3-16), including the insertion of A11;
  * Example D (Tables 17-19);
  * the simulation study on the frequency of rank reversal (Table 20);
  * the sensitivity analysis of Example D (Table 21).

Requirements: Python 3.9+, numpy, scipy (results in the paper obtained with Python 3.13, numpy 2.5, scipy 1.18).

Usage:  
        python topsis_rad_experiments.py            (everything; the simulation takes about one minute)
        python topsis_rad_experiments.py --quick    (skips the simulation and the sensitivity analysis)

All input data are defined in this file; no external file is needed. 

For reproducibility purposes, the random seed is fixed, so the output is identical on every run. 
"""

import sys, itertools
import numpy as np

# =============================================================================
# 1. Methods
# =============================================================================

def _rank(s):
    """Rank 1 = best (highest score); ties share the best rank (competition ranking)."""
    s = np.asarray(s, float)
    r = np.empty(len(s), int)
    for i in range(len(s)):
        r[i] = 1 + np.sum(s > s[i] + 1e-12)
    return r

def closeness(T, tpos, tneg):
    db = np.sqrt(((T - tpos) ** 2).sum(1))
    dw = np.sqrt(((T - tneg) ** 2).sum(1))
    cl = dw / (db + dw)
    return db, dw, cl

def topsis_vector(G, w, p):
    """Classical TOPSIS (Hwang & Yoon, 1981): vector normalisation, data-driven PIS/NIS."""
    G = np.asarray(G, float)
    w = np.asarray(w, float) / np.sum(w)
    p = np.asarray(p)
    T = G / np.sqrt((G ** 2).sum(0)) * w
    tpos = np.where(p == 1, T.max(0), T.min(0))
    tneg = np.where(p == 1, T.min(0), T.max(0))
    db, dw, cl = closeness(T, tpos, tneg)
    return db, dw, cl

def topsis_maxmin(G, w, p, lo=None, hi=None):
    """TOPSIS with max-min normalisation. lo/hi = None -> data-driven column extremes;
    lo/hi given -> fixed criterion domains (fixed-reference TOPSIS, Aires & Ferreira type)."""
    G = np.asarray(G, float)
    w = np.asarray(w, float) / np.sum(w)
    p = np.asarray(p)
    lo = G.min(0) if lo is None else np.asarray(lo, float)
    hi = G.max(0) if hi is None else np.asarray(hi, float)
    R = np.where(p == 1, (G - lo) / (hi - lo), (hi - G) / (hi - lo))
    T = R * w
    db, dw, cl = closeness(T, w, np.zeros_like(w))
    return db, dw, cl

def topsis_rad(G, w, p, vpl, dpl):
    """
        TOPSIS-RAD. Returns (feasible mask, R, T, D_ib, D_iw, S) with NaN rows for vetoed alternatives.
        Requires p_j*(d_j - v_j) > 0. 
        Feasible iff p_j*(g_ij - v_j) >= 0 for all j.
    """
    G = np.asarray(G, float)
    w = np.asarray(w, float) / np.sum(w)
    p = np.asarray(p)
    v = np.asarray(vpl, float)
    d = np.asarray(dpl, float)
    assert np.all(p * (d - v) > 0), "DPL must be strictly better than VPL on every criterion"
    feas = np.all(p * (G - v) >= 0, axis=1)
    Gd = np.where(p == 1, np.minimum(G, d), np.maximum(G, d))
    R = (Gd - v) / (d - v)
    T = R * w
    db, dw, S = closeness(T, w, np.zeros_like(w))
    for a in (R, T): 
        a[~feas] = np.nan
    db[~feas] = np.nan
    dw[~feas] = np.nan
    S[~feas] = np.nan
    return feas, R, T, db, dw, S

def surplus(G, w, p, vpl, dpl):
    """Weighted normalised surplus above the desired levels (optional tie-breaking index)."""
    G = np.asarray(G, float)
    w = np.asarray(w, float) / np.sum(w)
    p = np.asarray(p)
    v = np.asarray(vpl, float)
    d = np.asarray(dpl, float)
    return (w * np.maximum(0, p * (G - d)) / np.abs(d - v)).sum(1)

def rank_nan(S):
    r = np.full(len(S), 0)
    ok = ~np.isnan(S)
    r[ok] = _rank(S[ok])
    return r


# =============================================================================
# 2. Examples A, B and C (dataset of Table 3)
# =============================================================================

ALT = ['A%d' % i for i in range(1, 11)]
G_ABC = np.array([[68, 72, 74, 78], [62, 76, 78, 80], [58, 70, 150, 76], [80, 72, 99, 90], [90, 88, 47, 118],
                  [80, 64, 66, 70], [76, 68, 98, 105], [80, 50, 5, 41], [70, 74, 78, 75], [60, 68, 82, 81]], float)
W_ABC = np.full(4, 0.25)
P_ABC = np.ones(4, int)

A11 = np.array([25, 30, 25, 45.0])

def show(title, names, S, extra=None):
    r = rank_nan(S); print('\n' + title)
    for i, n in enumerate(names):
        tail = '' if extra is None else '   ' + extra[i]
        print('  %-4s %s  rank %s%s' % (n, '  vetoed' if np.isnan(S[i]) else '%.4f' % S[i], '-' if np.isnan(S[i]) else r[i], tail))
    print('  order: ' + ' > '.join(names[i] for i in np.argsort(-np.nan_to_num(S, nan=-1)) if not np.isnan(S[i])))

def show_RT(title, names, R, T):
    print('\n' + title + '\n        ' + '  '.join('R_c%d  ' % (j + 1) for j in range(R.shape[1])) + '  ' + '  '.join('T_c%d  ' % (j + 1) for j in range(T.shape[1])))
    for i, n in enumerate(names):
        if np.isnan(R[i]).all(): print('  %-4s  vetoed' % n)
        else: print('  %-4s ' % n + ' '.join('%.4f' % x for x in R[i]) + '  ' + ' '.join('%.4f' % x for x in T[i]))

def examples_abc():
    G, w, p = G_ABC, W_ABC, P_ABC
    R = (G - G.min(0)) / (G.max(0) - G.min(0))
    show_RT('Example A - normalised matrix R and weighted matrix T, data-driven Max-Min (Table 4)', ALT, R, R * w)
    db, dw, S = topsis_maxmin(G, w, p)
    
    show('Example A - TOPSIS, data-driven Max-Min normalisation (Table 6)', ALT, S, ['D_ib=%.4f D_iw=%.4f' % t for t in zip(db, dw)])
    show('Example A - original TOPSIS, vector normalisation (Table 7)', ALT, topsis_vector(G, w, p)[2])
    
    G9 = np.delete(G, 7, 0)
    show('Example A without A8 - TOPSIS, Max-Min recomputed (Table 11, row 2)', [a for a in ALT if a != 'A8'], topsis_maxmin(G9, w, p)[2])
    f, R, T, db, dw, S = topsis_rad(G, w, p, [20] * 4, [90, 88, 150, 118])
    
    show('Example B - TOPSIS-RAD, VPL=(20,20,20,20), DPL=(90,88,150,118) (Table 10)', ALT, S, ['D_ib=%.4f D_iw=%.4f' % t for t in zip(db, dw)])
    show_RT('Example B - normalised matrix R and weighted matrix T (Table 9)', ALT, R, T)
    f, R, T, db, dw, S = topsis_rad(G, w, p, [58, 50, 5, 41], [80] * 4)
    
    show('Example C - TOPSIS-RAD, VPL=(58,50,5,41), DPL=(80,80,80,80) (Table 14)', ALT, S, ['D_ib=%.4f D_iw=%.4f' % t for t in zip(db, dw)])
    show_RT('Example C - normalised matrix R and weighted matrix T (Table 13)', ALT, R, T)
    G11 = np.vstack([G, A11])
    N11 = ALT + ['A11']
    print('\n--- Insertion of A11 = (25, 30, 25, 45) (Table 16) ---')
    
    show('TOPSIS Max-Min, A + A11', N11, topsis_maxmin(G11, w, p)[2])
    show('TOPSIS vector, A + A11', N11, topsis_vector(G11, w, p)[2])
    
    for v, d in [([20] * 4, [90, 88, 150, 118]), ([58, 50, 5, 41], [80] * 4), ([20] * 4, [80] * 4)]:
        show('TOPSIS-RAD VPL=%s DPL=%s, A' % (v, d), ALT, topsis_rad(G, w, p, v, d)[5])
        show('TOPSIS-RAD VPL=%s DPL=%s, A + A11' % (v, d), N11, topsis_rad(G11, w, p, v, d)[5])
    
    print('\nA11 is dominated by:', [ALT[i] for i in range(10) if np.all(G[i] >= A11) and np.any(G[i] > A11)])
    

    Gr = np.array([[100.0] * 3, [90.0] * 3, [80.0] * 3])
    wr = np.full(3, 1 / 3)
    pr = np.ones(3, int)
    print('\nThree dominated alternatives, VPL=(50,50,50): DPL=(80,80,80) -> S =', topsis_rad(Gr, wr, pr, [50] * 3, [80] * 3)[5].round(4), ' E =', surplus(Gr, wr, pr, [50] * 3, [80] * 3).round(3),
          '; DPL=(100,100,100) -> S =', topsis_rad(Gr, wr, pr, [50] * 3, [100] * 3)[5].round(4))


# =============================================================================
# 3. Example D (hypothetical supplier-selection data; mixed directions, unequal weights)
# =============================================================================

SUP = ['S%d' % i for i in range(1, 9)]
G_D = np.array([[100, 12, 99.0, 96, 75], [92, 20, 97.5, 90, 60], [70, 28, 96.0, 87, 55], [110, 7, 99.8, 99, 90],
                [97, 10, 99.4, 97, 82], [65, 15, 93.0, 92, 70], [118, 5, 99.9, 99.5, 95], [105, 14, 98.5, 94, 65]], float)
W_D = np.array([0.30, 0.15, 0.25, 0.20, 0.10])
P_D = np.array([-1, -1, 1, 1, 1])
VPL_D = np.array([120, 30, 95, 85, 50.0])
DPL_D = np.array([60, 10, 99.5, 98, 80.0])
LO_D = np.zeros(5)
HI_D = np.array([150, 60, 100, 100, 100.0])
S9 = np.array([125, 4, 99.0, 84, 45.0])

def example_d():
    f, R, T, db, dw, S = topsis_rad(G_D, W_D, P_D, VPL_D, DPL_D)
    E = surplus(G_D, W_D, P_D, VPL_D, DPL_D)
    
    print('\n--- Example D ---\nNormalised matrix R (Table 18):')
    print(np.round(R, 4))
    
    show('Example D - TOPSIS-RAD (Table 18)', SUP, S, ['D_ib=%.4f D_iw=%.4f E=%.4f' % t for t in zip(db, dw, E)])
    for label, GG, nm in [('A', G_D, SUP), ('A + S9', np.vstack([G_D, S9]), SUP + ['S9'])]:
        show('TOPSIS vector, ' + label, nm, topsis_vector(GG, W_D, P_D)[2])
        show('TOPSIS Max-Min, ' + label, nm, topsis_maxmin(GG, W_D, P_D)[2])
        show('Fixed-reference TOPSIS, ' + label, nm, topsis_maxmin(GG, W_D, P_D, LO_D, HI_D)[2])
        show('TOPSIS-RAD, ' + label, nm, topsis_rad(GG, W_D, P_D, VPL_D, DPL_D)[5])

# =============================================================================
# 4. Simulation: frequency of rank reversal (Table 20)
# =============================================================================
def pair_reversal(S_before, S_after):
    ok = ~np.isnan(S_before) & ~np.isnan(S_after)
    a, b = S_before[ok], S_after[ok]
    if len(a) < 2: return False, False
    da = np.sign(np.round(a[:, None] - a[None, :], 12))
    db = np.sign(np.round(b[:, None] - b[None, :], 12))
    return bool(np.any(da != db)), bool(np.argmax(a) != np.argmax(b))

def four_methods(G, w, p):
    n = G.shape[1]; v = np.where(p == 1, 10.0, 90.0)
    d = np.where(p == 1, 80.0, 20.0)
    return {'vector': topsis_vector(G, w, p)[2], 'maxmin': topsis_maxmin(G, w, p)[2],
            'fixed': topsis_maxmin(G, w, p, np.zeros(n), np.full(n, 100.0))[2], 'rad': topsis_rad(G, w, p, v, d)[5]}

def simulation(reps=10000, seed=2026):
    rng = np.random.default_rng(seed)
    print('\n--- Simulation (Table 20): % of replications; columns = addition any pair / addition best / removal any pair / removal best ---')
    for m, n in itertools.product([5, 10, 20, 50], [3, 5, 8]):
        cnt = {k: np.zeros(4) for k in ['vector', 'maxmin', 'fixed', 'rad']}
        for _ in range(reps):
            G = rng.uniform(0, 100, (m + 1, n))
            w = rng.dirichlet(np.ones(n))
            p = rng.choice([-1, 1], n)
            full = four_methods(G, w, p)
            base = four_methods(G[:m], w, p)
            k = rng.integers(m)
            red = four_methods(np.delete(G[:m], k, 0), w, p)
            for key in cnt:
                a1, t1 = pair_reversal(base[key], full[key][:m])
                a2, t2 = pair_reversal(np.delete(base[key], k), red[key])
                cnt[key] += [a1, t1, a2, t2]
        print('m=%2d n=%d  ' % (m, n) + '  '.join('%s %s' % (k, (100 * v / reps).round(1).tolist()) for k, v in cnt.items()), flush=True)

# =============================================================================
# 5. Sensitivity analysis of Example D (Table 21)
# =============================================================================
def sensitivity(reps=10000, seed=2026):
    from scipy.stats import kendalltau
    rng = np.random.default_rng(seed)
    span = np.abs(DPL_D - VPL_D)
    f0, _, _, _, _, S0 = topsis_rad(G_D, W_D, P_D, VPL_D, DPL_D)
    top0 = np.nanargmax(S0)
    print('\n--- Sensitivity (Table 21): same best (%), same qualified set (%), same complete ranking (%), mean Kendall tau ---')
    for what in ['VPL', 'DPL', 'VPL+DPL', 'weights', 'all']:
        for delta in [0.05, 0.10, 0.20]:
            same_top = same_set = same_rank = 0; taus = []
            for _ in range(reps):
                v, d, w = VPL_D.copy(), DPL_D.copy(), W_D.copy()
                if what in ('VPL', 'VPL+DPL', 'all'): v = VPL_D + rng.uniform(-delta, delta, 5) * span
                if what in ('DPL', 'VPL+DPL', 'all'): d = DPL_D + rng.uniform(-delta, delta, 5) * span
                if what in ('weights', 'all'): 
                    w = W_D * (1 + rng.uniform(-delta, delta, 5))
                    w /= w.sum()
                f, _, _, _, _, S = topsis_rad(G_D, w, P_D, v, d)
                ok = f & f0
                same_set += np.array_equal(f, f0)
                same_top += bool(f[top0] and np.nanargmax(S) == top0)
                t = kendalltau(S0[ok], S[ok]).statistic
                taus.append(t)
                same_rank += bool(np.array_equal(f, f0) and t > 1 - 1e-12)
            print('%-8s +-%2d%%  %5.1f  %5.1f  %5.1f  %.3f' % (what, delta * 100, 100 * same_top / reps, 100 * same_set / reps, 100 * same_rank / reps, np.mean(taus)), flush=True)

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    examples_abc()
    example_d()
    if '--quick' not in sys.argv:
        simulation()
        sensitivity()
