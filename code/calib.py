"""Simulator-versus-human comparison for the six non-RL conditions of McKee et al. (2023) across kappa.

Five type-based planners (static, random recommendations, neutral, maximum connectivity, encouragement) run
in the batched PyTorch simulator; cooperative clustering (a per-node rule) runs in the NumPy reference
implementation (sim_numpy.py). Human targets are computed locally from the OSF group-outcome files.
Also evaluated: the propensity-matched variants (deltas.json) and the payoff-biased imitation variant.
usage: python calib.py deltas.json out.json [n_games]
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import torch
import rdp
import sim_numpy


def np_job(args):
    cond, seed, n = args
    r = sim_numpy.run('clustering', n_games=n, seed=seed, kappa=cond.get('kappa', 1.0), w_imit=cond.get('w_imit', 0.0))
    return [(x['final'], x['mean'], x['capital']) for x in r]


def np_clustering(cond, n, workers=8):
    if cond.get('delta'):
        return None                       # the NumPy reference has no intercept-shift option
    chunks = [(cond, 100 + i, n // workers) for i in range(workers)]
    with Pool(workers) as p:
        res = sum(p.map(np_job, chunks), [])
    a = np.array(res)
    return dict(coop_final=float(a[:, 0].mean()), coop_all=float(a[:, 1].mean()), cap=float(a[:, 2].mean()),
                coop_final_se=float(a[:, 0].std() / np.sqrt(len(a))), n=len(a))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = json.load(open(sys.argv[1]))
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    KG = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    CONDS = [dict(kappa=k) for k in KG]
    CONDS += [dict(kappa=k, delta=D[f'slopeM_{k}']) for k in [0.5, 0.75]]
    CONDS += [dict(kappa=1.0, delta=D[f'levelM_{k}']) for k in [0.5, 0.75]]
    CONDS += [dict(kappa=1.0, w_imit=0.3), dict(kappa=0.5, w_imit=0.3)]
    PL = {'static': rdp.FixedPolicy(np.zeros((14, 3, 2))), 'neutral': rdp.FixedPolicy(rdp.fixed_tables()['neutral']),
          'encouragement': rdp.FixedPolicy(rdp.enc_table()), 'random': rdp.AllPairsPolicy('random', 0.30),
          'maxconn': rdp.AllPairsPolicy('maxconn')}






    out = []
    for cond in CONDS:
        for name, pol in PL.items():
            ev = rdp.evaluate(pol.to(dev), cond=cond, n_games=N, device=dev, seed=2468)
            coop_all = float(np.mean(ev['traj']))
            out.append(dict(cond=cond, planner=name, coop_final=ev['coop_final'], coop_final_se=ev['coop_final_se'],
                            coop_all=coop_all, cap=ev['cap'], iso_final=ev['iso_final'], impl='torch'))
        c = np_clustering(cond, max(N // 4, 800))
        if c:
            out.append(dict(cond=cond, planner='clustering', impl='numpy', **c))
        print(json.dumps(cond), flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=0)


if __name__ == '__main__':
    main()
