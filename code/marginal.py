"""Value of keeping versus removing a cooperator-defector (C-D) tie.

States: games are run under a planner up to its decision after round t (t in {2, 7, 13}); the recommended
changes are applied, and then, before round t+1 is played, one existing C-D tie is drawn uniformly per game
(games without a C-D tie are skipped, so each game contributes at most one tie and the bootstrap over ties
is a bootstrap over games). All differences are 'keep minus remove'.

One-step value: exact change in the two endpoints' round-(t+1) cooperation probabilities under the bot model,
with and without the capital gate (cooperation is feasible only if c*k <= capital), decomposed (Shapley,
two-factor) into the degree term (beta_1 * k) and the neighbourhood term (kappa * (beta_2 x_n + beta_3 x_r)).
The tie only enters the logits of its two endpoints, so the group-level one-step change equals the sum.
Break-even kappa: the defector multiplier at which the mean one-step value is zero, for the fixed state set.
Multi-step value: two copies of each state (tie kept / removed) are simulated to the end of the game under
the same planner with common random numbers; we report the change in the number of cooperators in round
t+1 (a check against the analytic one-step value), in the final round, summed over rounds t+1..15, the
change in total payoff over rounds t+1..15 and in minimum final capital.
usage: python marginal.py rundir out_prefix [n_games]
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import rdp
from envtorch import Game, RL, PHI_DEL_C, PHI_DEL_D, PHI_ADD_C, PHI_ADD_D

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rundir, prefix = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
KGRID = np.round(np.arange(0.0, 1.5001, 0.005), 3)


def net_update(g, R):
    """First half of Game.step: apply accepted recommendations, advance the clock, do not play."""
    R = R.bool()
    e = g.pair_vals(g.A) > 0.5
    ai, aj = g.a[:, g.iu], g.a[:, g.ju]
    u = g._rand(g.B, g.m) < 0.5
    a_ref = torch.where(u, ai, aj) > 0.5
    phi = torch.where(e, torch.where(a_ref, torch.full_like(ai, PHI_DEL_C), torch.full_like(ai, PHI_DEL_D)),
                      torch.where(a_ref, torch.full_like(ai, PHI_ADD_C), torch.full_like(ai, PHI_ADD_D)))
    acc = g._rand(g.B, g.m) < phi
    g.A = g.set_pairs((e ^ (R & acc)).float())
    g.t += 1


def load_table(name):
    r = json.load(open(os.path.join(rundir, name + '.json')))
    return rdp.FixedPolicy(r['table']).to(dev)


def states(pol, kappa, t_dec, seed):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    g = Game(N, dev, kappa=kappa, gen=gen)
    obs = g.reset()
    for t in range(1, t_dec + 1):
        logit, _ = pol(g, obs['A'], obs['a'], obs['d'], t)
        act = torch.rand(logit.shape, device=dev, generator=gen) < torch.sigmoid(logit)
        if t < t_dec:
            g.step(act)
            obs = g.obs()
        else:
            net_update(g, act)
    a = g.a
    e = g.pair_vals(g.A) > 0.5
    cd = e & ((a[:, g.iu] + a[:, g.ju]) == 1)
    has = cd.any(-1)
    w = torch.where(cd, torch.rand(cd.shape, device=dev, generator=gen), torch.full_like(cd, -1.0, dtype=torch.float))
    pidx = w.argmax(-1)
    i, j = g.iu[pidx], g.ju[pidx]
    ci = torch.where(a.gather(1, i[:, None]).squeeze(1) > 0.5, i, j)
    di = torch.where(ci == i, j, i)
    keep = has.nonzero().squeeze(1)
    return g, keep, ci, di


def one_step(g, keep, ci, di, kappa, cost=0.05):
    b = keep
    A, a, d, th = g.A[b], g.a[b], g.d[b], g.theta[b]
    deg = A.sum(-1)
    xn = torch.bmm(A, a.unsqueeze(-1)).squeeze(-1)
    c, dd = ci[b], di[b]
    gat = lambda x, idx: x.gather(1, idx[:, None]).squeeze(1)
    res = {}
    for role, idx, k in (('C', c, 1.0), ('D', dd, kappa)):
        k1, x1, t_ = gat(deg, idx), gat(xn, idx), gat(th, idx)
        dcap = gat(d, idx)
        if role == 'C':
            k0, x0 = k1 - 1, x1                      # removing a defecting neighbour
        else:
            k0, x0 = k1 - 1, x1 - 1                  # removing a cooperating neighbour
        r1 = x1 / k1.clamp(min=1)
        r0 = torch.where(k0 > 0, x0 / k0.clamp(min=1), torch.zeros_like(x0))
        x0 = torch.where(k0 > 0, x0, torch.zeros_like(x0))
        base0 = RL[0] + RL[1] * k0 + t_
        Ldeg = RL[1] * (k1 - k0)
        Lnb = k * (RL[2] * (x1 - x0) + RL[3] * (r1 - r0))
        L0 = base0 + k * (RL[2] * x0 + RL[3] * r0)
        s = torch.sigmoid
        p1, p0 = s(L0 + Ldeg + Lnb), s(L0)
        res['d' + role] = (p1 - p0)
        res['d' + role + '_deg'] = 0.5 * ((s(L0 + Ldeg) - p0) + (p1 - s(L0 + Lnb)))
        res['d' + role + '_nb'] = 0.5 * ((s(L0 + Lnb) - p0) + (p1 - s(L0 + Ldeg)))
        f1 = (cost * k1 <= dcap + 1e-6).float()
        f0 = (cost * k0 <= dcap + 1e-6).float()
        res['d' + role + '_gated'] = p1 * f1 - p0 * f0
        res['k_' + role] = k1
        res['xr_' + role] = r1
    res['dTot'] = res['dC'] + res['dD']
    res['dTot_gated'] = res['dC_gated'] + res['dD_gated']
    return {k: v.double().cpu().numpy() for k, v in res.items()}


def kappa_curve(g, keep, ci, di):
    return np.stack([one_step(g, keep, ci, di, float(k))['dTot'] for k in KGRID], 1)   # (ties, grid)


def break_even(M, rng, nb=2000):
    """Smallest kappa with non-negative mean tie value, interpolated on KGRID."""
    def be(m):
        if m[0] >= 0:
            return float(KGRID[0])
        idx = np.flatnonzero((m[:-1] < 0) & (m[1:] >= 0))
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return KGRID[i] + (0 - m[i]) * (KGRID[i + 1] - KGRID[i]) / (m[i + 1] - m[i])
    est = be(M.mean(0))
    boots = [be(M[rng.integers(0, len(M), len(M))].mean(0)) for _ in range(nb)]
    return est, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


@torch.no_grad()
def rollout(pol, g, keep, ci, di, kappa, remove, seed):
    b = keep
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    h = Game(len(b), dev, kappa=kappa, gen=gen)
    A = g.A[b].clone()
    if remove:
        r = torch.arange(len(b), device=dev)
        A[r, ci[b], di[b]] = 0.0
        A[r, di[b], ci[b]] = 0.0
    h.hist = {k: [] for k in g.hist}
    h.load_state(A, g.theta[b], g.a[b], g.pay[b], g.d[b], g.t)
    d_before = h.d.clone()
    h._play()
    n_next = h.a.sum(-1)
    for t in range(g.t, rdp.NDEC + 1):
        logit, _ = pol(h, h.A, h.a, h.d, t)
        act = torch.rand(logit.shape, device=dev, generator=gen) < torch.sigmoid(logit)
        h.step(act)
    coop = torch.stack(h.hist['coop'], 0) * h.n
    return dict(n_next=n_next, n_final=coop[-1], n_cum=coop.sum(0), pay=(h.d - d_before).sum(-1),
                minc=h.d.min(-1).values)


def ci95(x, rng, nb=2000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rng = np.random.default_rng(7)
PLANNERS = {'encouragement': lambda k: rdp.FixedPolicy(rdp.enc_table()).to(dev)}
for tag in ['welfare', 'coopmean', 'e3']:
    PLANNERS['es_' + tag] = (lambda tg: (lambda k: load_table(f'table_{tg}_k{k}_s0')))(tag)
rows, ties = [], []
seed = 1
for pname, mk in PLANNERS.items():
    for kap in [1.0, 0.75, 0.5]:
        pol = mk(kap)
        for t_dec in [2, 7, 13]:
            seed += 1
            g, keep, ci, di = states(pol, kap, t_dec, 1000 + seed)
            o = one_step(g, keep, ci, di, kap)
            M = kappa_curve(g, keep, ci, di)
            be, be_lo, be_hi = break_even(M, rng)
            ro = {}
            outs = [rollout(pol, g, keep, ci, di, kap, rm, 5000 + seed) for rm in (False, True)]
            for k in outs[0]:
                ro[k] = (outs[0][k] - outs[1][k]).double().cpu().numpy()
            row = dict(planner=pname, kappa=kap, t=t_dec, n_ties=len(keep), frac_games_with_cd=len(keep) / N,
                       break_even=be, break_even_lo=be_lo, break_even_hi=be_hi,
                       frac_pos=float((o['dTot'] > 0).mean()))
            for k in ['dC', 'dD', 'dTot', 'dC_deg', 'dC_nb', 'dD_deg', 'dD_nb', 'dTot_gated']:
                m, lo, hi = ci95(o[k], rng)
                row.update({k: m, k + '_lo': lo, k + '_hi': hi})
            for k, v in ro.items():
                m, lo, hi = ci95(v, rng)
                row.update({'ms_' + k: m, 'ms_' + k + '_lo': lo, 'ms_' + k + '_hi': hi})
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if not k.endswith(('_lo', '_hi'))}), flush=True)
            if pname == 'encouragement':
                sel = rng.choice(len(keep), min(5000, len(keep)), replace=False)
                T = pd.DataFrame({k: o[k][sel] for k in ['dC', 'dD', 'dTot', 'dTot_gated', 'dD_deg', 'dD_nb',
                                                          'dC_deg', 'dC_nb', 'k_C', 'k_D', 'xr_C', 'xr_D']})
                T['ms_n_final'] = ro['n_final'][sel]
                T['ms_pay'] = ro['pay'][sel]
                T['planner'], T['kappa'], T['t'] = pname, kap, t_dec
                ties.append(T)
pd.DataFrame(rows).to_csv(prefix + '_summary.csv', index=False)
pd.concat(ties).to_csv(prefix + '_ties.csv.gz', index=False)
