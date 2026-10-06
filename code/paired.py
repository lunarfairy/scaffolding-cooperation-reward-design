"""Paired statistics for regret and objective compatibility, from eval_all.py output.

For every simulator condition c and objective o: planner families (kind, label) trained in c plus all
hand-specified planners compete. The best family b*(o, c) is selected on seed B; its seed-A mean is J*.
For each family f we report on seed A: raw shortfall J* - J_f, its Monte Carlo standard error from paired
per-game differences (family per-game outcome = mean over training seeds, common random numbers), a
95% interval from a bootstrap over training seeds (resampling seeds within f and within b*), and the
normalised regret (J* - J_f) / (J* - J_static), unclipped.
usage: python paired.py evaldir out.csv
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

S_W = 0.66
OBJS = ['e3', 'welfare', 'omega0.75', 'omega0.5', 'omega0.25', 'desert', 'mix0.25', 'mix0.5', 'mix0.75',
        'coopmean', 'coopfinal', 'sustain', 'rawls']


def objs_from(g):
    d = {'e3': g['e3'], 'welfare': g['welfare_ret'], 'coopmean': g['coop_ret'] / 14, 'coopfinal': g['coop_final'],
         'rawls': g['rawls'], 'desert': g['desert'], 'sustain': g['sustain']}
    for w in [0.25, 0.5, 0.75]:
        d[f'omega{w}'] = g['desert'] + w * (g['welfare_ret'] - g['desert'])
    for a in [0.25, 0.5, 0.75]:
        d[f'mix{a}'] = a * g['coop_ret'] + (1 - a) * g['welfare_ret'] / S_W
    return d


evaldir, outcsv = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(evaldir, 'evals_*.jsonl'))) for l in open(f)]
E = pd.DataFrame(rows)
E = E[(E.kind == 'fixed') | (E.train_cond == E.eval_cond)]
rng = np.random.default_rng(0)
out = []
for c, sub in E.groupby('eval_cond'):
    # seed-B family means for selection
    B = sub[sub.seed == 24680]
    selB = {}
    for (kind, fam), g in B.groupby(['kind', 'family']):
        js = [objs_from({k: r[k] for k in ['e3', 'welfare_ret', 'coop_ret', 'coop_final', 'rawls', 'desert', 'sustain']})
              for _, r in g.iterrows()]
        selB[(kind, fam)] = {o: float(np.mean([j[o] for j in js])) for o in OBJS}
    A = sub[sub.seed == 12345]
    fams = {}
    for (kind, fam), g in A.groupby(['kind', 'family']):
        arrs = []
        for nm in g.name:
            fn = os.path.join(evaldir, 'games', f'{nm}__{c}.npz')
            if os.path.exists(fn):
                z = np.load(fn)
                arrs.append(objs_from({k: z[k].astype(np.float64) for k in z.files}))
        if arrs:
            fams[(kind, fam)] = arrs
    if ('fixed', 'static') not in fams:
        continue
    for o in OBJS:
        cand = [k for k in fams if k in selB]
        best = max(cand, key=lambda k: selB[k][o])
        Jb_seeds = np.array([a[o] for a in fams[best]])          # (S_b, games)
        Jstat = fams[('fixed', 'static')][0][o].mean()
        Jstar = Jb_seeds.mean()
        bestA = max(fams, key=lambda k: np.mean([a[o].mean() for a in fams[k]]))
        for k, arrs in fams.items():
            Jf_seeds = np.array([a[o] for a in arrs])
            diff = Jb_seeds.mean(0) - Jf_seeds.mean(0)
            se_mc = diff.std() / np.sqrt(len(diff))
            mb, mf = Jb_seeds.mean(1), Jf_seeds.mean(1)
            boot = []
            for _ in range(2000):
                bb = mb[rng.integers(0, len(mb), len(mb))].mean()
                ff = mf[rng.integers(0, len(mf), len(mf))].mean()
                boot.append(bb - ff)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            if k == best:
                lo, hi = 0.0, 0.0
            den = Jstar - Jstat
            out.append(dict(cond=c, obj=o, kind=k[0], family=k[1], n_seeds=len(arrs), J=float(mf.mean()),
                            J_seed_sd=float(mf.std(ddof=1)) if len(mf) > 1 else np.nan,
                            Jstar=float(Jstar), best_kind=best[0], best_family=best[1],
                            bestA_kind=bestA[0], bestA_family=bestA[1], Jstatic=float(Jstat),
                            shortfall=float(Jstar - mf.mean()), se_mc=float(se_mc), ci_lo=float(lo), ci_hi=float(hi),
                            regret=float((Jstar - mf.mean()) / den) if den > 0 else np.nan))
    print(c, len(fams), flush=True)
pd.DataFrame(out).to_csv(outcsv, index=False)
