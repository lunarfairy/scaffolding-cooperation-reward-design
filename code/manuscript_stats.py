"""Generate manuscript statistics without figure-generation dependencies.

Usage: python manuscript_stats.py RESULTS_DIR OSF_DIR OUT_DIR
Inputs: RESULTS_DIR/evals/evals_*.jsonl, paired.csv and calib.json;
        OSF_DIR/{baseline,evaluation,validation}_group_outcomes_data.csv.
Outputs: numbers_rev.json, calibration_mae.csv and best_found_k0.5.csv.

Scientific calculations are extracted from the authors' revision analysis2.py,
stats_rev.py, figs_rev.py and make_figures.py. Plot styling is omitted.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd


S_W = 0.66


OBJS = ['e3', 'welfare', 'omega0.75', 'omega0.5', 'omega0.25', 'desert', 'mix0.25', 'mix0.5', 'mix0.75',
        'coopmean', 'coopfinal', 'sustain', 'rawls']


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


def own(E, cond, kind, fam, seed=12345):
    return E[(E.eval_cond == cond) & (E.seed == seed) & E.own & (E.kind == kind) & (E.family == fam)]


def phase(sub, pair, status, rounds, which='rec_rates'):
    num = den = 0.0
    for _, r in sub.iterrows():
        opp = np.array(r['rec_opp'])[:, pair, status][rounds]
        num += (np.array(r[which])[:, pair, status][rounds] * opp).sum(); den += opp.sum()
    return num / den if den else np.nan


FAMS = ['e3', 'welfare', 'desert', 'rawls', 'coopmean', 'coopfinal', 'sustain']


METS = ['coop_final', 'cap', 'rawls', 'iso_final', 'coop_conn_final', 'gap', 'welfare_ret', 'nchg', 'forced',
        'contrib', 'sustain', 'conv', 'subsidy']


LATE, EARLY = list(range(7, 14)), list(range(0, 4))


def fam_stats(E, cond, kind, f):
    sub = own(E, cond, kind, f)
    if not len(sub):
        return None
    d = {'n': int(len(sub))}
    for m in METS + ['J_' + o for o in OBJS]:
        x = sub[m].astype(float).values
        d[m] = float(x.mean())
        d[m + '_sd'] = float(x.std(ddof=1)) if len(x) > 1 else None
    d['cd_add_early'] = phase(sub, 1, 0, EARLY)
    d['cd_add_late'] = phase(sub, 1, 0, LATE)
    d['cd_cut_early'] = phase(sub, 1, 1, EARLY)
    d['cd_cut_late'] = phase(sub, 1, 1, LATE)
    d['cd_cut_mid'] = phase(sub, 1, 1, list(range(4, 9)))
    d['cd_cutchg_late'] = phase(sub, 1, 1, LATE, 'chg_rates')
    d['cd_addchg_early'] = phase(sub, 1, 0, EARLY, 'chg_rates')
    d['cc_add_all'] = phase(sub, 2, 0, list(range(14)))
    d['dd_cut_all'] = phase(sub, 0, 1, list(range(14)))
    per = [phase(sub.iloc[[i]], 1, 1, LATE) for i in range(len(sub))]
    d['cd_cut_late_seed_range'] = [float(min(per)), float(max(per))]
    return d


def compute_numbers(E, P, out):
    N = {'own': {}}
    conds = sorted(E.eval_cond.unique())
    for c in conds:
        N['own'][c] = {}
        for kind in ['table', 'gnn', 'table_r', 'fixed']:
            fams = sorted(E[(E.eval_cond == c) & E.own & (E.kind == kind)].family.unique())
            for f in fams:
                s = fam_stats(E, c, kind, f)
                if s:
                    N['own'][c][f'{kind}:{f}'] = s
    # frozen transfer
    T = E[(E.seed == 12345) & E.kind.isin(['table', 'gnn']) & ~E.train_cond.str.contains('_') & ~E.eval_cond.str.contains('_')]
    tr = T.groupby(['kind', 'family', 'train_cond', 'eval_cond'])[['coop_final', 'cap', 'rawls']].agg(['mean', 'std', 'count'])
    N['transfer'] = {f'{k}:{f}|{a}->{b}': {m: float(tr.loc[(k, f, a, b), (m, 'mean')]) for m in ['coop_final', 'cap', 'rawls']}
                     | {'sd': float(tr.loc[(k, f, a, b), ('coop_final', 'std')]) if tr.loc[(k, f, a, b), ('coop_final', 'count')] > 1 else None,
                        'n': int(tr.loc[(k, f, a, b), ('coop_final', 'count')])}
                     for (k, f, a, b) in tr.index}
    # imitation transfer (w_train x w_test, kappa = 1)
    W = E[(E.seed == 12345) & E.kind.isin(['table', 'gnn']) & E.train_cond.isin(['k1', 'k1_w_imit0.3']) & E.eval_cond.isin(['k1', 'k1_w_imit0.3'])]
    wt = W.groupby(['kind', 'family', 'train_cond', 'eval_cond']).coop_final.mean()
    N['w_transfer'] = {f'{k}:{f}|{a}->{b}': float(v) for (k, f, a, b), v in wt.items()}
    # ES vs GNN agreement on final cooperation (principal objectives, own condition)
    pairs = []
    for c in ['k0.5', 'k0.75', 'k1']:
        for f in FAMS:
            a, b = N['own'][c].get(f'table:{f}'), N['own'][c].get(f'gnn:{f}')
            if a and b:
                pairs.append((c, f, a['coop_final'], b['coop_final']))
    if pairs:
        x = np.array([p[2] for p in pairs]); y = np.array([p[3] for p in pairs])
        from scipy.stats import spearmanr, pearsonr
        N['es_gnn'] = dict(n=len(pairs), pearson=float(pearsonr(x, y)[0]), spearman=float(spearmanr(x, y)[0]),
                           mae=float(np.abs(x - y).mean()), max_abs=float(np.abs(x - y).max()),
                           per_kappa={c: dict(pearson=float(pearsonr([p[2] for p in pairs if p[0] == c], [p[3] for p in pairs if p[0] == c])[0]),
                                              mae=float(np.mean([abs(p[2] - p[3]) for p in pairs if p[0] == c])))
                                      for c in ['k0.5', 'k0.75', 'k1'] if sum(p[0] == c for p in pairs) > 2},
                           pairs=pairs)
    # regret / compatibility
    if P is not None:
        R = {}
        for c in ['k1', 'k0.75', 'k0.5', 'k1_w_imit0.3']:
            sub = P[P.cond == c]
            R[c] = {f'{k}:{f}': {o: dict(regret=float(g.regret.iloc[0]), shortfall=float(g.shortfall.iloc[0]),
                                          se_mc=float(g.se_mc.iloc[0]), ci=[float(g.ci_lo.iloc[0]), float(g.ci_hi.iloc[0])])
                                  for o, g in gg.groupby('obj')}
                    for (k, f), gg in sub.groupby(['kind', 'family'])}
            R[c]['_best'] = {o: f"{g.best_kind.iloc[0]}:{g.best_family.iloc[0]}" for o, g in sub.groupby('obj')}
            R[c]['_bestA'] = {o: f"{g.bestA_kind.iloc[0]}:{g.bestA_family.iloc[0]}" for o, g in sub.groupby('obj')}
            R[c]['_Jstar'] = {o: float(g.Jstar.iloc[0]) for o, g in sub.groupby('obj')}
        N['regret'] = R
        # compatibility of each ES planner trained on objective o at kappa = 1 (and GNN)
        comp = {}
        for c in ['k1', 'k0.5']:
            sub = P[P.cond == c]
            for kind in ['table', 'gnn']:
                for o in OBJS:
                    g = sub[(sub.kind == kind) & (sub.family == o)]
                    if not len(g):
                        continue
                    g = g.set_index('obj')
                    own_r = g.loc[o, 'regret']
                    order = g.regret.sort_values()
                    tol = 2 * g.se_mc.max() / max(1e-9, (g.Jstar - g.Jstatic).abs().min())
                    within = [ob for ob in order.index if g.loc[ob, 'regret'] - order.iloc[0] <= 0.005]
                    comp[f'{c}|{kind}:{o}'] = dict(own=float(own_r), min_obj=order.index[0], min=float(order.iloc[0]),
                                                  strict=bool(order.index[0] == o), rank=int(list(order.index).index(o)) + 1,
                                                  within_0005=within, own_minus_min=float(own_r - order.iloc[0]))
        N['compat'] = comp
    with open(out, 'w') as output:
        json.dump(N, output, indent=1, default=float)
    return N


def calibration_mae(cal, OSF):
    mp = {'Coop. clustering': 'clustering', 'Encourag. planner': 'encouragement', 'Max. connectivity': 'maxconn',
          'Neutral planner': 'neutral', 'Random rec.': 'random', 'Static network': 'static', 'GraphNet planner': 'graphnet'}
    G = pd.concat([pd.read_csv(os.path.join(OSF, f'{f}_group_outcomes_data.csv')) for f in ['baseline', 'evaluation', 'validation']])
    human = G[G.Round == G.Round.max()].groupby('Condition').Mean_Cooperation.mean().rename(index=mp).to_dict()
    P5 = ['static', 'random', 'encouragement', 'neutral', 'maxconn']
    rows = []
    for c, g in cal.groupby(cal.cond.apply(lambda c: json.dumps(c, sort_keys=True))):
        cd = json.loads(c); g = g.set_index('planner')
        kind = 'kappa' if set(cd) == {'kappa'} else ('level' if cd.get('delta', 0) < 0 else ('slope' if cd.get('delta', 0) > 0 else 'imit'))
        rows.append(dict(kappa=cd['kappa'], delta=cd.get('delta', 0), w=cd.get('w_imit', 0), kind=kind,
                         mae_final=np.mean([abs(g.loc[p, 'coop_final'] - human[p]) for p in P5])))
    CM = pd.DataFrame(rows)
    return CM


def best_found(N):
    O = N['own']['k0.5']
    brow = []
    for o in ['e3', 'welfare', 'desert', 'rawls', 'coopmean', 'coopfinal', 'sustain']:
        J = 'J_' + o
        cand = {'ES': O.get(f'table:{o}', {}).get(J),
                'ES multi-start': max([O[k][J] for k in O if k.startswith(f'table:{o}-init')] or [np.nan]),
                'GNN': O.get(f'gnn:{o}', {}).get(J), 'GNN joint': O.get(f'gnn:{o}-joint', {}).get(J),
                'hand rule': max(O[k][J] for k in O if k.startswith('fixed:'))}
        best = max(v[J] for v in O.values())
        for k, v in cand.items():
            if v is not None and not np.isnan(v):
                brow.append(dict(obj=o, src=k, J=v, short_pct=100 * (best - v) / abs(best)))
    BEST = pd.DataFrame(brow)

    return BEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', help='Directory containing evals/, paired.csv and calib.json')
    parser.add_argument('osf_dir', help='Directory containing the three released human group-outcome CSVs')
    parser.add_argument('out_dir', help='Directory for generated JSON and CSV statistics')
    args = parser.parse_args()
    R, OSF, OUT = args.results_dir, args.osf_dir, args.out_dir
    os.makedirs(OUT, exist_ok=True)
    E = load_evals(os.path.join(R, 'evals')).drop_duplicates(['name', 'eval_cond', 'seed'])
    P = pd.read_csv(os.path.join(R, 'paired.csv'))
    with open(os.path.join(R, 'calib.json')) as source:
        cal = pd.DataFrame(json.load(source))
    N = compute_numbers(E, P, os.path.join(OUT, 'numbers_rev.json'))
    calibration_mae(cal, OSF).to_csv(os.path.join(OUT, 'calibration_mae.csv'), index=False)
    best_found(N).to_csv(os.path.join(OUT, 'best_found_k0.5.csv'), index=False)
    print('statistics written to', OUT)


if __name__ == '__main__':
    main()
