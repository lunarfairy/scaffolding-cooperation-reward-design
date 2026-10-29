"""Generate the five main figures, four Extended Data figures and summary statistics.

usage: python make_figures.py RESULTS_DIR OSF_DIR OUT_DIR
  RESULTS_DIR contains evals/ (evals_*.jsonl), paired.csv, marg_summary.csv, coef_draws.csv, calib.json,
              runs_es/ and runs_gnn/ (per-run JSON with training logs)
  OSF_DIR     contains the released group-outcome files of McKee et al. (osf.io/8ahkg)
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis2
import figs_rev
import stats_rev

R, OSF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
E = analysis2.load_evals(os.path.join(R, 'evals')).drop_duplicates(['name', 'eval_cond', 'seed'])
P = pd.read_csv(os.path.join(R, 'paired.csv'))
marg = pd.read_csv(os.path.join(R, 'marg_summary.csv'))
coef = pd.read_csv(os.path.join(R, 'coef_draws.csv'))
cal = pd.DataFrame(json.load(open(os.path.join(R, 'calib.json'))))
N = stats_rev.main(E, P, os.path.join(OUT, 'numbers_rev.json'))

# human targets
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
sty = {'level': ('s', '#d6604d', 'κ = 1, lower intercept'), 'imit': ('v', '#1b7837', 'κ = 1, imitation'), 'kappa': ('o', '#2166ac', '')}
sel = CM[(CM.kind == 'kappa') | ((CM.kind == 'level') & (CM.delta > -1.5)) | ((CM.kind == 'imit') & (CM.kappa == 1))].copy()
sel['marker'] = sel.kind.map(lambda k: sty[k][0]); sel['color'] = sel.kind.map(lambda k: sty[k][1]); sel['label'] = sel.kind.map(lambda k: sty[k][2])

runs_es = {json.load(open(f))['cfg']['name']: json.load(open(f))['log'] for f in glob.glob(os.path.join(R, 'runs_es', '*.json'))}
runs_gnn = {json.load(open(f))['cfg']['name']: json.load(open(f))['log'] for f in glob.glob(os.path.join(R, 'runs_gnn', '*.json'))}
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

for ext in ['pdf', 'png']:
    figs_rev.fig1(E, coef, marg, sel, os.path.join(OUT, f'fig1_mechanism.{ext}')); plt.close('all')
    figs_rev.fig2(E, os.path.join(OUT, f'fig2_schedules.{ext}')); plt.close('all')
    figs_rev.fig3(E, os.path.join(OUT, f'fig3_outcomes.{ext}')); plt.close('all')
    figs_rev.fig4(E, os.path.join(OUT, f'fig4_robustness.{ext}')); plt.close('all')
    figs_rev.fig5(P, E, os.path.join(OUT, f'fig5_regret.{ext}')); plt.close('all')
    figs_rev.ed_sched(E, os.path.join(OUT, f'ed_fig1_schedules.{ext}')); plt.close('all')
    figs_rev.ed_optim(E, runs_es, runs_gnn, BEST, os.path.join(OUT, f'ed_fig2_optimization.{ext}')); plt.close('all')
    figs_rev.ed_tie(marg, coef, os.path.join(OUT, f'ed_fig3_tie_value.{ext}')); plt.close('all')
    figs_rev.ed_calib(cal, human, os.path.join(OUT, f'ed_fig4_calibration.{ext}')); plt.close('all')
BEST.to_csv(os.path.join(OUT, 'best_found_k0.5.csv'), index=False)
CM.to_csv(os.path.join(OUT, 'calibration_mae.csv'), index=False)
print('figures written to', OUT)
