"""Re-evaluate every trained and hand-specified planner on common evaluation scenario sets.

Seed A (12345) is the reporting set and seed B (24680) the selection set used to choose the best planner
for each objective (avoiding winner's-curse bias). Within a seed and simulator condition all planners share
initial networks, dispositions and the random-number stream (common random numbers).
Each trained planner is evaluated in its own training condition (A and B, 20,000 games each; per-game
outcomes of A saved for paired statistics) and, if trained in a 'base' condition (kappa grid, optionally
payoff-biased imitation with w = 0.3), as a frozen policy in every base test condition (A, 10,000 games).
Hand-specified planners are evaluated in every condition that occurs (A and B, 20,000 games).

usage: python eval_all.py launch W gpus outdir rundir1 [rundir2 ...]
       python eval_all.py worker i W outdir rundirs(comma-separated)
"""
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
import rdp

SEED_A, SEED_B = 12345, 24680
N_OWN = int(os.environ.get('RDP_N_OWN', 20000))
N_CROSS = int(os.environ.get('RDP_N_CROSS', 10000))
BASE_TEST = [dict(kappa=k) for k in (0.5, 0.625, 0.75, 0.875, 1.0)] + [dict(kappa=1.0, w_imit=0.3),
                                                                     dict(kappa=0.5, w_imit=0.3)]
MISSPEC = ('delta', 'ben', 'cost', 'mu_th', 'temp', 'imit')


def norm_cond(c):
    c = {k: v for k, v in rdp.game_kw(c).items()}
    c['kappa'] = float(c.get('kappa', 1.0))
    if not c.get('w_imit'):
        c.pop('w_imit', None)
    return c


def ckey(c):
    c = norm_cond(c)
    s = f"k{c['kappa']:g}"
    for k in ('w_imit', 'temp', 'imit', 'delta', 'ben', 'cost', 'mu_th'):
        if k in c:
            v = c[k]
            s += f"_{k}{v:.4g}" if isinstance(v, float) else f"_{k}{v}"
    return s


def is_base(c):
    c = norm_cond(c)
    return not any(k in c for k in MISSPEC) and c.get('w_imit', 0.0) in (0.0, 0.3)


def family(cfg):
    suf = ''
    if cfg.get('init'):
        suf += '-init' + cfg['init'][:4]
    if cfg.get('freeze'):
        suf += '-' + cfg['freeze'].replace('_', '')
    if cfg.get('ppo_mode') == 'joint':
        suf += '-joint'
    return cfg['policy'], cfg['tag'] + suf


def weight_file(run_path):
    adjacent = run_path[:-5] + '.pt'
    if os.path.exists(adjacent):
        return adjacent
    return os.path.join(os.path.dirname(os.path.dirname(run_path)), 'weights_gnn',
                        os.path.basename(adjacent))


def policies(rundirs):
    out = []
    for d in rundirs:
        for f in sorted(glob.glob(os.path.join(d, '*.json'))):
            r = json.load(open(f))
            if 'cfg' not in r:
                continue
            cfg = r['cfg']
            if cfg['policy'] == 'gnn' and not os.path.exists(weight_file(f)):
                continue
            out.append(dict(name=cfg['name'], path=f, cfg=cfg))
    return out


def fixed_policies():
    P = {k: v for k, v in rdp.fixed_tables().items() if k not in ('switch0', 'switch14')}
    return [dict(name='fixed_' + k, fixed=k, cfg=dict(policy='fixed', tag=k)) for k in P]


def tasks(pols):
    T, conds = [], {}
    for p in pols:
        c = norm_cond(p['cfg'])
        conds[ckey(c)] = c
        T.append((p, c, SEED_A, N_OWN, True))
        T.append((p, c, SEED_B, N_OWN, False))
        if is_base(c):
            for tc in BASE_TEST:
                if ckey(tc) != ckey(c):
                    T.append((p, norm_cond(tc), SEED_A, N_CROSS, False))
    for tc in BASE_TEST:
        conds[ckey(tc)] = norm_cond(tc)
    for fp in fixed_policies():
        for k, c in sorted(conds.items()):
            T.append((fp, c, SEED_A, N_OWN, True))
            T.append((fp, c, SEED_B, N_OWN, False))
    return T


def load(p, dev):
    if 'fixed' in p:
        return rdp.FixedPolicy(rdp.fixed_tables()[p['fixed']]).to(dev)
    r = json.load(open(p['path']))
    if 'table' in r:
        return rdp.FixedPolicy(r['table']).to(dev)
    pol = rdp.GNNPolicy()
    pol.load_state_dict(torch.load(weight_file(p['path']), map_location='cpu'))
    return pol.to(dev).eval()


def worker(i, W, outdir, rundirs):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pols = policies(rundirs)
    T = tasks(pols)
    mine = [t for j, t in enumerate(T) if j % W == i]
    os.makedirs(os.path.join(outdir, 'games'), exist_ok=True)
    tag = os.environ.get('RDP_EVAL_TAG', 'f')
    fn = os.path.join(outdir, f'evals_{tag}{i:02d}.jsonl')
    done = set()
    for ff in glob.glob(os.path.join(outdir, 'evals_*.jsonl')):
        for line in open(ff):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r['name'], r['eval_cond'], r['seed']))
    f = open(fn, 'a')
    cache = {}
    t0 = time.time()
    for n_done, (p, c, seed, n, keep) in enumerate(mine):
        key = (p['name'], ckey(c), seed)
        if key in done:
            continue
        if p['name'] not in cache:
            cache = {p['name']: load(p, dev)}
        ev = rdp.evaluate(cache[p['name']], cond=c, n_games=n, device=dev, seed=seed, keep_games=keep)
        g = ev.pop('_games', None)
        kind, fam = family(p['cfg'])
        row = dict(name=p['name'], kind=kind, family=fam, seed_train=p['cfg'].get('seed', 0),
                   train_cond=ckey(p['cfg']) if kind != 'fixed' else 'any', eval_cond=ckey(c), seed=seed, n=n)
        row.update({k: v for k, v in ev.items() if k not in ('rec_counts', 'cond')})
        row['rec_opp'] = np.array(ev['rec_counts'])[..., 0].tolist()
        f.write(json.dumps(row) + '\n')
        f.flush()
        if keep and g is not None:
            np.savez_compressed(os.path.join(outdir, 'games', f"{p['name']}__{ckey(c)}.npz"),
                                **{k: v.astype(np.float32) for k, v in g.items()})
        if n_done % 50 == 0:
            print(i, n_done, len(mine), '%.0fs' % (time.time() - t0), flush=True)
    f.close()
    print('worker', i, 'done', time.time() - t0, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'launch':
        W, gpus, outdir, rundirs = int(sys.argv[2]), sys.argv[3].split(','), sys.argv[4], sys.argv[5:]
        os.makedirs(outdir, exist_ok=True)
        print('tasks', len(tasks(policies(rundirs))), flush=True)
        procs = []
        for i in range(W):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)], OMP_NUM_THREADS='2')
            lf = open(os.path.join(outdir, f'worker_{i:02d}.log'), 'w')
            procs.append(subprocess.Popen([sys.executable, 'eval_all.py', 'worker', str(i), str(W), outdir,
                                           ','.join(rundirs)], stdout=lf, stderr=subprocess.STDOUT, env=env))
        for p in procs:
            p.wait()
        print('rc', [p.returncode for p in procs], flush=True)
    else:
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(','))
