"""Train one planner under one objective, save its parameters and a quick own-condition evaluation.

usage: python run_one.py '<json cfg>' outdir [deltas.json]
cfg keys: name, obj, policy (gnn|table|table_r), seed, iters, batch, P, gamma, omega, alpha, K,
          simulator keys (kappa, w_imit, temp, delta, ben, cost, mu_th, imit), ES keys (npair, games, sigma,
          lr, init, freeze), PPO keys (lr, ent, epochs, minibatches, ppo_mode).
String values of 'delta' (e.g. 'slopeM_0.5') and 'P' (e.g. 'match_inc') are resolved from deltas.json.
All reported results are produced afterwards by eval_all.py on common evaluation scenario sets.
"""
import atexit
import json
import os
import sys
import time
import torch
import rdp

cfg = json.loads(sys.argv[1])
out = sys.argv[2]
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    D = json.load(open(sys.argv[3]))
    for key in ('delta', 'P'):
        if isinstance(cfg.get(key), str):
            cfg[key + '_key'] = cfg[key]
            cfg[key] = float(D[cfg[key]]) * float(cfg.get(key + '_scale', 1.0))
os.makedirs(out, exist_ok=True)
# a second worker pool may train the same configuration; the first to start owns it (lock file)
_js = os.path.join(out, cfg['name'] + '.json'); _lk = _js + '.lock'
if os.path.exists(_js):
    sys.exit(0)
if os.path.exists(_lk):
    for _ in range(480):
        if os.path.exists(_js):
            sys.exit(0)
        time.sleep(30)
    sys.exit(1)
open(_lk, 'w').write(str(os.getpid()))


def release_lock():
    if os.path.exists(_lk):
        os.remove(_lk)


atexit.register(release_lock)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.environ.get('RDP_THREADS', '2')))
t0 = time.time()
es = cfg['policy'] in ('table', 'table_r')
pol, log = rdp.train_es(cfg, device=dev) if es else rdp.train(cfg, device=dev)
t_train = time.time() - t0
pol.eval()
res = dict(cfg=cfg, log=log, train_sec=t_train, device=dev)
ev = rdp.evaluate(pol, cond=rdp.game_kw(cfg), n_games=cfg.get('eval_games', 10000), device=dev)
res['eval'] = {'own': {k: v for k, v in ev.items() if not k.startswith('_')}}
if es:
    res['table'] = rdp.table_probs(pol)
res['total_sec'] = time.time() - t0
with open(os.path.join(out, cfg['name'] + '.json'), 'w') as f:
    json.dump(res, f)
torch.save(pol.state_dict(), os.path.join(out, cfg['name'] + '.pt'))
print(cfg['name'], 'train %.0fs' % t_train, 'coop_final %.3f cap %.3f' % (ev['coop_final'], ev['cap']))
