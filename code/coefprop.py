"""Uncertainty and specification sensitivity of the one-step C-D tie value and its break-even kappa.

States are generated exactly as in marginal.py (distilled schedule, published simulator, decision rounds
2, 7, 13; one uniformly drawn C-D tie per game, 20,000 games). Holding these states fixed we
 (i) draw the three later-round slope coefficients (degree, x_n, x_r) independently from normal
     distributions centred on the published values, with standard errors derived from the bootstrap 95%
     intervals released with the original analysis (baseline_cooperation_glmer_eff_data.csv, converted
     from standardised to raw units with the predictor SDs of the released baseline data);
 (ii) vary where kappa enters: 'base' (x_n and x_r of last-round defectors), 'r_only', 'xn_only', and
     'all' (neighbourhood terms of both endpoints);
and solve for the break-even kappa at which the mean one-step change in the endpoints' cooperation
probabilities is zero. Also reports the one-step value stratified by the defector's degree and x_r.
usage: python coefprop.py out_prefix [n_games] [n_draws]
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import rdp
from envtorch import Game, RL, PHI_DEL_C, PHI_DEL_D, PHI_ADD_C, PHI_ADD_D

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
prefix = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
ND = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SE = dict(deg=0.0581, xn=0.0690, xr=0.4067)          # raw-unit SEs (see docstring)


def net_update(g, R):
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


def features(t_dec, seed, kappa=1.0):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    pol = rdp.FixedPolicy(rdp.enc_table()).to(dev)
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
    p = w.argmax(-1)
    i, j = g.iu[p], g.ju[p]
    ci = torch.where(a.gather(1, i[:, None]).squeeze(1) > 0.5, i, j)
    di = torch.where(ci == i, j, i)
    b = has.nonzero().squeeze(1)
    deg = g.A.sum(-1)
    xn = torch.bmm(g.A, a.unsqueeze(-1)).squeeze(-1)
    gat = lambda x, idx: x[b].gather(1, idx[b][:, None]).squeeze(1)
    return dict(kC=gat(deg, ci), xC=gat(xn, ci), thC=gat(g.theta, ci), kD=gat(deg, di), xD=gat(xn, di),
                thD=gat(g.theta, di))


def endpoint(k1, x1, th, removed_coop, bdeg, bxn, bxr, kap_n, kap_r):
    k0 = k1 - 1
    x0 = torch.where(k0 > 0, x1 - (1.0 if removed_coop else 0.0), torch.zeros_like(x1))
    r1 = x1 / k1.clamp(min=1)
    r0 = torch.where(k0 > 0, x0 / k0.clamp(min=1), torch.zeros_like(x0))
    L1 = RL[0] + bdeg * k1 + kap_n * bxn * x1 + kap_r * bxr * r1 + th
    L0 = RL[0] + bdeg * k0 + kap_n * bxn * x0 + kap_r * bxr * r0 + th
    return torch.sigmoid(L1) - torch.sigmoid(L0)


def dtot(F, kap, b=(RL[1], RL[2], RL[3]), spec='base'):
    bd, bx, br = b
    one = 1.0
    kd_n = kap if spec in ('base', 'xn_only', 'all') else one
    kd_r = kap if spec in ('base', 'r_only', 'all') else one
    kc = kap if spec == 'all' else one
    dC = endpoint(F['kC'], F['xC'], F['thC'], False, bd, bx, br, kc, kc)
    dD = endpoint(F['kD'], F['xD'], F['thD'], True, bd, bx, br, kd_n, kd_r)
    return dC + dD


KG = np.round(np.arange(0.0, 1.5001, 0.005), 3)


def break_even(F, b, spec):
    """Smallest kappa in [0, 1.5] at which the mean one-step value turns from negative to non-negative
    (linear interpolation on a 0.005 grid). Returns 0 if the value is already non-negative at kappa = 0 and
    NaN if it never becomes non-negative. The value is not monotone in kappa: for large kappa defectors'
    cooperation probabilities saturate and the defector-side gain shrinks again."""
    kg = torch.as_tensor(KG, dtype=torch.float32, device=dev)[:, None]
    m = dtot({k: v[None, :] for k, v in F.items()}, kg, b, spec).mean(1).cpu().numpy()
    if m[0] >= 0:
        return 0.0
    idx = np.flatnonzero((m[:-1] < 0) & (m[1:] >= 0))
    if len(idx) == 0:
        return float('nan')
    i = idx[0]
    return float(KG[i] + (0 - m[i]) * (KG[i + 1] - KG[i]) / (m[i + 1] - m[i]))


def curve(F, spec):
    return [float(dtot(F, float(k), spec=spec).mean()) for k in KG[::10]]


rng = np.random.default_rng(11)
rows, strat = [], []
for t_dec, seed in [(2, 31), (7, 32), (13, 33)]:
    F = features(t_dec, seed)
    for spec in ['base', 'r_only', 'xn_only', 'all']:
        rows.append(dict(t=t_dec, spec=spec, draw='point', break_even=break_even(F, (RL[1], RL[2], RL[3]), spec),
                         n=len(F['kC']), curve=json.dumps(curve(F, spec))))
    for d in range(ND):
        b = (RL[1] + SE['deg'] * rng.standard_normal(), RL[2] + SE['xn'] * rng.standard_normal(),
             RL[3] + SE['xr'] * rng.standard_normal())
        rows.append(dict(t=t_dec, spec='base', draw=d, break_even=break_even(F, b, 'base'), b_deg=b[0], b_xn=b[1],
                         b_xr=b[2], n=len(F['kC']),
                         value_k1=float(dtot(F, 1.0, b, 'base').mean()),       # mean one-step value at kappa = 1
                         value_k05=float(dtot(F, 0.5, b, 'base').mean())))     # and at kappa = 0.5
    for kap in [1.0, 0.75, 0.5]:
        v = dtot(F, kap).cpu().numpy()
        kD = F['kD'].cpu().numpy()
        rD = (F['xD'] / F['kD'].clamp(min=1)).cpu().numpy()
        df = pd.DataFrame(dict(v=v, kbin=pd.cut(kD, [0, 3, 6, 9, 16]), rbin=pd.cut(rD, [-0.01, 1 / 3, 2 / 3, 1.0])))
        for (kb, rb), g in df.groupby(['kbin', 'rbin'], observed=True):
            strat.append(dict(t=t_dec, kappa=kap, deg_bin=str(kb), xr_bin=str(rb), n=len(g), mean=g.v.mean(),
                              se=g.v.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan))
    print(t_dec, json.dumps([r for r in rows if r['t'] == t_dec and r['draw'] == 'point']), flush=True)
R = pd.DataFrame(rows)
R.to_csv(prefix + '_draws.csv', index=False)
pd.DataFrame(strat).to_csv(prefix + '_strata.csv', index=False)
S = R[R.draw != 'point'].groupby('t').break_even.describe(percentiles=[0.025, 0.5, 0.975])
print(S)
