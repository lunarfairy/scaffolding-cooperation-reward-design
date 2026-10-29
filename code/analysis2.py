"""Aggregate the revision evaluation output (eval_all.py, paired.py) into tidy tables.

All learned-planner summaries are means over training seeds of seed-A (reporting set) evaluations;
uncertainty is the s.d. across training seeds and a t-based 95% interval of the seed mean.
Phase summaries of recommendation rates are opportunity-weighted (pooled over games, rounds and seeds):
rate = sum(recommendations) / sum(opportunities). Realised change rates are accepted changes per opportunity.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

S_W = 0.66
OBJS = ['e3', 'welfare', 'omega0.75', 'omega0.5', 'omega0.25', 'desert', 'mix0.25', 'mix0.5', 'mix0.75',
        'coopmean', 'coopfinal', 'sustain', 'rawls']
PHASES = {'1-4': range(0, 4), '5-9': range(4, 9), '10-14': range(9, 14), '8-14': range(7, 14)}
SCALARS = ['coop_final', 'coop_mean', 'cap', 'rawls', 'cap_p10', 'gini', 'gap', 'dens', 'sustain', 'e3', 'desert',
           'welfare_ret', 'coop_ret', 'nrec', 'rej', 'nchg', 'subsidy', 'benD_share', 'conv', 'degC', 'degD', 'capC',
           'capD', 'iso_final', 'coop_conn_final', 'forced', 'contrib', 'imit_flip']


def objvals(r):
    d = {'e3': r['e3'], 'welfare': r['welfare_ret'], 'coopmean': r['coop_ret'] / 14, 'coopfinal': r['coop_final'],
         'rawls': r['rawls'], 'desert': r['desert'], 'sustain': r['sustain']}
    for w in [0.25, 0.5, 0.75]:
        d[f'omega{w}'] = r['desert'] + w * (r['welfare_ret'] - r['desert'])
    for a in [0.25, 0.5, 0.75]:
        d[f'mix{a}'] = a * r['coop_ret'] + (1 - a) * r['welfare_ret'] / S_W
    return d


def load_evals(evaldir):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(evaldir, 'evals_*.jsonl'))) for l in open(f)]
    E = pd.DataFrame(rows)
    J = pd.DataFrame([objvals(r) for _, r in E.iterrows()]).add_prefix('J_')
    E = pd.concat([E.reset_index(drop=True), J], axis=1)
    E['own'] = (E.kind == 'fixed') | (E.train_cond == E.eval_cond)
    return E


def phase_rates(sub, pair=1, status=1, which='rec'):
    """Opportunity-weighted rate for pair type (0 DD, 1 CD, 2 CC) and status (0 absent/add, 1 present/delete)."""
    out = {}
    for ph, idx in PHASES.items():
        num = den = 0.0
        for _, r in sub.iterrows():
            opp = np.array(r['rec_opp'])[:, pair, status]
            rate = np.array(r['rec_rates' if which == 'rec' else 'chg_rates'])[:, pair, status]
            num += (rate[list(idx)] * opp[list(idx)]).sum()
            den += opp[list(idx)].sum()
        out[ph] = num / den if den > 0 else np.nan
    return out


def summarize(E, cond, seed=12345, kinds=None, metrics=SCALARS):
    sub = E[(E.eval_cond == cond) & (E.seed == seed) & E.own]
    if kinds:
        sub = sub[sub.kind.isin(kinds)]
    rows = []
    for (kind, fam), g in sub.groupby(['kind', 'family']):
        row = dict(cond=cond, kind=kind, family=fam, n_seeds=len(g))
        for m in metrics + ['J_' + o for o in OBJS]:
            x = g[m].astype(float).values
            row[m] = x.mean()
            row[m + '_sd'] = x.std(ddof=1) if len(x) > 1 else np.nan
            row[m + '_ci'] = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        for st, nm in [(0, 'add'), (1, 'del')]:
            for ph, v in phase_rates(g, 1, st).items():
                row[f'cd_{nm}_{ph}'] = v
            for ph, v in phase_rates(g, 1, st, 'chg').items():
                row[f'cdchg_{nm}_{ph}'] = v
        row['cc_add_all'] = phase_rates(g, 2, 0)['1-4'] * 0 + np.nanmean([phase_rates(g, 2, 0)[p] for p in ['1-4', '5-9', '10-14']])
        row['dd_del_all'] = np.nanmean([phase_rates(g, 0, 1)[p] for p in ['1-4', '5-9', '10-14']])
        rows.append(row)
    return pd.DataFrame(rows)


def transfer(E, kinds=('table', 'gnn'), fams=None, metric='coop_final'):
    """Frozen-policy transfer: train condition x test condition, base kappa grid (w = 0), seed A."""
    sub = E[(E.seed == 12345) & E.kind.isin(kinds) & ~E.train_cond.str.contains('_') & ~E.eval_cond.str.contains('_')]
    if fams:
        sub = sub[sub.family.isin(fams)]
    g = sub.groupby(['kind', 'family', 'train_cond', 'eval_cond'])[metric].agg(['mean', 'std', 'count']).reset_index()
    return g
