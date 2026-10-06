"""Reference quantities computed once before the experiment grid (writes deltas.json).

1. Propensity-matched perturbations. Reference states are last-round defectors in rounds 2-15 of games run
   under the distilled (encouragement) schedule with the published simulator (kappa = 1). For kappa < 1 we
   solve for an additive defector intercept delta such that the mean cooperation probability of these
   reference defectors equals a target:
     slope-matched  ('slopeM_k'):  kappa = k with delta > 0, matching the kappa = 1 mean propensity
                                   (changes only the slope on neighbourhood cooperation);
     level-only     ('levelM_k'):  kappa = 1 with delta < 0, matching the kappa = k mean propensity
                                   (changes only the level, keeping the published slope).
2. Reward scales used to match the rejection penalty across reward accountings: mean per-decision
   capital level, payoff increment and cooperation rate under the same reference run.
usage: python propensity.py out.json
"""
import json
import sys
import torch
import rdp
from envtorch import Game, RL

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = torch.Generator(device=dev)
gen.manual_seed(4242)
pol = rdp.FixedPolicy(rdp.enc_table()).to(dev)
B = 8000
g = Game(B, dev, gen=gen)
obs = g.reset()
feats = []
lev, inc, coop = [], [], []
for t in range(1, rdp.NDEC + 1):
    logit, _ = pol(g, obs['A'], obs['a'], obs['d'], t)
    act = torch.rand(logit.shape, device=dev, generator=gen) < torch.sigmoid(logit)
    a_prev = g.a.clone()
    info = g.step(act)
    deg = g.deg
    xn = torch.bmm(g.A, a_prev.unsqueeze(-1)).squeeze(-1)
    xr = xn / deg.clamp(min=1)
    msk = a_prev < 0.5
    feats.append(torch.stack([deg[msk], xn[msk], xr[msk], g.theta[msk]], -1))
    lev.append(float(g.d.mean()))
    inc.append(float(g.pay.mean()))
    coop.append(float(g.a.mean()))
    obs = g.obs()
F = torch.cat(feats, 0).double()
base = RL[0] + RL[1] * F[:, 0] + F[:, 3]
nb = RL[2] * F[:, 1] + RL[3] * F[:, 2]


def pmean(k, delta):
    return float(torch.sigmoid(base + k * nb + delta).mean())


def solve(k, target):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pmean(k, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


out = dict(n_ref_defectors=int(F.shape[0]), p_ref_k1=pmean(1.0, 0.0))
for k in [0.5, 0.625, 0.75, 0.875]:
    out[f'p_ref_k{k}'] = pmean(k, 0.0)
    out[f'slopeM_{k}'] = solve(k, pmean(1.0, 0.0))
    out[f'levelM_{k}'] = solve(1.0, pmean(k, 0.0))
S_level, S_inc, S_coop = sum(lev) / len(lev), sum(inc) / len(inc), sum(coop) / len(coop)
out.update(S_level=S_level, S_inc=S_inc, S_coop=S_coop, match_inc=S_inc / S_level, match_coop=S_coop / S_level)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
