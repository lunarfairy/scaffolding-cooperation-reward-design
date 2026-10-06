"""Cross-check of the batched PyTorch simulator against the independent NumPy reference implementation.
usage: python refcheck.py out.json [n_games]"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import torch
import rdp
import sim_numpy


def job(args):
    name, kap, w, seed, n = args
    r = sim_numpy.run(name, n_games=n, seed=seed, kappa=kap, w_imit=w)
    return [(x['final'], x['capital'], x['mean']) for x in r]


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    PL = {'static': rdp.FixedPolicy(np.zeros((14, 3, 2))), 'encouragement': rdp.FixedPolicy(rdp.enc_table()),
          'neutral': rdp.FixedPolicy(rdp.fixed_tables()['neutral']), 'random': rdp.AllPairsPolicy('random', 0.30),
          'maxconn': rdp.AllPairsPolicy('maxconn')}




    rows = []
    for kap, w in [(1.0, 0.0), (0.5, 0.0), (1.0, 0.3)]:
        for name, pol in PL.items():
            ev = rdp.evaluate(pol.to(dev), cond=dict(kappa=kap, w_imit=w), n_games=N, device=dev, seed=97531)
            with Pool(8) as p:
                res = np.array(sum(p.map(job, [(name, kap, w, 500 + i, N // 8) for i in range(8)]), []))
            rows.append(dict(planner=name, kappa=kap, w_imit=w,
                             torch_final=ev['coop_final'], torch_final_se=ev['coop_final_se'], torch_cap=ev['cap'],
                             numpy_final=float(res[:, 0].mean()), numpy_final_se=float(res[:, 0].std() / np.sqrt(len(res))),
                             numpy_cap=float(res[:, 1].mean()), torch_all=float(np.mean(ev['traj'])),
                             numpy_all=float(res[:, 2].mean()), n=N))
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(sys.argv[1], 'w'), indent=0)


if __name__ == '__main__':
    main()
