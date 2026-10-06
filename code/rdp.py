"""Reward design for AI social planners: policies, objectives, PPO training and evaluation.

Policy classes
  gnn   : message-passing graph network, per-pair Bernoulli 'recommend a change' outputs
  table : one logit per (round, pair type CC/CD/DD, edge status), i.e. the distilled form used by
          McKee et al. for their encouragement planner (SI Tables 7-9)
  table_r: state-augmented table, additionally indexed by a 3-level bin of the cooperative-neighbour
          fraction x_r of the pair's defecting endpoint (C-D pairs) or the endpoint mean (C-C, D-D)
  fixed : hand-specified probability tables (baselines, switch family)
Objectives (per planner decision t = 1..14, reward observed after round t+1)
  e3        : mean capital level - P * rejected/120 (McKee SI E3; P = 1, gamma = 0.99). The penalty is
              subtracted (it is a cost); cfg['P'] replaces, not adds to, the default.
  welfare   : mean payoff of the round (sum = final mean capital - planner-independent round 1)
  omega     : desert-weighted welfare, benefits to defectors weighted by omega (omega=1 -> welfare)
  coop_mean : cooperation rate of the round
  coop_final: cooperation rate in the final round (terminal)
  rawls     : minimum final capital (terminal)
  sustain   : mean cooperation over K rounds after the planner withdraws, network frozen (terminal)
  mix       : alpha * coop + (1 - alpha) * payoff / S_W
"""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from envtorch import Game, BEN, COST, gini_t

NDEC = 14
# encouragement planner (GraphNet distillation, SI Tables 7-9): rows t = 1..14, cols (add, delete)
ENC_CC = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, .010], [1, .010],
          [1, .011], [1, .028], [.991, .035], [.954, .073], [1, .108]]
ENC_CD = [[.993, .048], [.973, .029], [.914, .145], [.791, .213], [.644, .318], [.594, .508],
          [.463, .608], [.429, .745], [.366, .802], [.372, .753], [.361, .741], [.371, .774],
          [.328, .706], [.408, .722]]
ENC_DD = [[0.0, 1.0]] * 14
NEUTRAL = [[.891, .119], [.841, .054], [.656, .084], [.642, .102], [.608, .117], [.549, .204],
           [.545, .215], [.538, .224], [.520, .239], [.504, .213], [.532, .215], [.518, .237],
           [.529, .232], [.522, .317]]
S_W = 0.66   # per-round payoff / per-round cooperation under the encouragement planner, kappa=1 (Methods)


GAME_KEYS = ('kappa', 'w_imit', 'temp', 'delta', 'ben', 'cost', 'mu_th', 'imit')


def game_kw(cfg):
    """Simulator keyword arguments from a config / condition dict (missing keys -> published model)."""
    return {k: cfg[k] for k in GAME_KEYS if k in cfg and cfg[k] is not None}


def enc_table():
    P = np.zeros((NDEC, 3, 2))           # [t, type(0 DD,1 CD,2 CC), status(0 absent->add, 1 present->delete)]
    P[:, 2] = ENC_CC
    P[:, 1] = ENC_CD
    P[:, 0] = ENC_DD
    return P


def switch_table(s, cc=(1.0, 0.0), dd=(0.0, 1.0)):
    P = np.zeros((NDEC, 3, 2))
    P[:, 2] = cc
    P[:, 0] = dd
    for t in range(1, NDEC + 1):
        P[t - 1, 1] = (1.0, 0.0) if t <= s else (0.0, 1.0)
    return P


def fixed_tables():
    tabs = {'static': np.zeros((NDEC, 3, 2)), 'encouragement': enc_table()}
    nt = np.zeros((NDEC, 3, 2))
    for k in range(3):
        nt[:, k] = NEUTRAL
    tabs['neutral'] = nt
    tabs['exclusion'] = switch_table(0)
    tabs['conciliation'] = switch_table(14)
    for s in range(0, 15):
        tabs[f'switch{s}'] = switch_table(s)
    return tabs


# ======================================================================== features
def pair_feats(game, A, a):
    e = A[:, game.iu, game.ju]
    typ = (a[:, game.iu] + a[:, game.ju]).long()        # 0 DD, 1 CD, 2 CC
    return e, typ


def pair_bins(game, A, a):
    """3-level bin of x_r: defecting endpoint for C-D pairs, endpoint mean otherwise."""
    deg = A.sum(-1)
    xr = torch.bmm(A, a.unsqueeze(-1)).squeeze(-1) / deg.clamp(min=1)
    xi, xj = xr[:, game.iu], xr[:, game.ju]
    ai = a[:, game.iu]
    aj = a[:, game.ju]
    cd = (ai + aj) == 1
    xdef = torch.where(ai < 0.5, xi, xj)
    x = torch.where(cd, xdef, 0.5 * (xi + xj))
    return (x >= 1 / 3).long() + (x >= 2 / 3).long()


def global_feats(game, A, a, d, t):
    e, typ = pair_feats(game, A, a)
    m = game.m
    tt = tfrac(t, a)
    g = torch.stack([tt, a.mean(-1), e.mean(-1), d.mean(-1) - 1.0,
                     (e * (typ == 2)).sum(-1) / m, (e * (typ == 1)).sum(-1) / m, (e * (typ == 0)).sum(-1) / m], -1)
    return g


def tfrac(t, a):
    """Round index (int or (B,) tensor, 1..14) -> (B,) float in [0, 1]."""
    if torch.is_tensor(t):
        return (t.to(a.dtype) - 1) / (NDEC - 1)
    return torch.full_like(a[:, 0], (t - 1) / (NDEC - 1))


def tidx(t, a):
    if torch.is_tensor(t):
        return (t - 1).long()
    return torch.full_like(a[:, 0], t - 1).long()


def mlp(i, h, o, act=nn.Tanh):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


class GNNPolicy(nn.Module):
    kind = 'gnn'

    def __init__(self, H=64, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(5, H), nn.Tanh())
        self.msg = nn.ModuleList([nn.Linear(H, H) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(4 * H + 1, H), nn.Tanh(), nn.Linear(H, H))
                                  for _ in range(layers)])
        self.edge = mlp(2 * H + 5, H, 1)
        self.val = mlp(2 * H + 7, H, 1)

    def forward(self, game, A, a, d, t):
        deg = A.sum(-1)
        xr = torch.bmm(A, a.unsqueeze(-1)).squeeze(-1) / deg.clamp(min=1)
        tt = tfrac(t, a)[:, None].expand_as(a)
        h = self.inp(torch.stack([a, d - 1.0, deg / 15.0, xr, tt], -1))
        for M, U in zip(self.msg, self.upd):
            mh = torch.tanh(M(h))
            s = torch.bmm(A, mh)
            mean = s / deg.clamp(min=1).unsqueeze(-1)
            pool = h.mean(1, keepdim=True).expand_as(h)
            h = torch.tanh(h + U(torch.cat([h, mean, s / 15.0, pool, tt.unsqueeze(-1)], -1)))
        hi, hj = h[:, game.iu], h[:, game.ju]
        e, typ = pair_feats(game, A, a)
        oh = torch.nn.functional.one_hot(typ, 3).float()
        tp = tt[:, :1].expand_as(e)
        z = torch.cat([hi + hj, hi * hj, e.unsqueeze(-1), oh, tp.unsqueeze(-1)], -1)
        logit = self.edge(z).squeeze(-1)
        g = global_feats(game, A, a, d, t)
        v = self.val(torch.cat([h.mean(1), h.max(1).values, g], -1)).squeeze(-1)
        return logit, v


class TablePolicy(nn.Module):
    kind = 'table'

    def __init__(self, H=64, init=0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((NDEC, 3, 2), float(init)))
        self.val = mlp(7, H, 1)

    def forward(self, game, A, a, d, t):
        e, typ = pair_feats(game, A, a)
        ti = tidx(t, a)[:, None].expand_as(typ)
        logit = self.logit[ti, typ, e.long()]
        v = self.val(global_feats(game, A, a, d, t)).squeeze(-1)
        return logit, v


class TablePolicyR(nn.Module):
    kind = 'table_r'

    def __init__(self, H=64, init=0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((NDEC, 3, 2, 3), float(init)))

    def forward(self, game, A, a, d, t):
        e, typ = pair_feats(game, A, a)
        ti = tidx(t, a)[:, None].expand_as(typ)
        return self.logit[ti, typ, e.long(), pair_bins(game, A, a)], torch.zeros_like(a[:, 0])


class FixedPolicy(nn.Module):
    """Probability table (14,3,2) or state-augmented (14,3,2,3); tables saved as probabilities."""
    kind = 'fixed'

    def __init__(self, P):
        super().__init__()
        P = torch.as_tensor(np.asarray(P), dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
        self.register_buffer('L', (torch.log(P) - torch.log1p(-P)).float())

    def forward(self, game, A, a, d, t):
        e, typ = pair_feats(game, A, a)
        ti = tidx(t, a)[:, None].expand_as(typ)
        if self.L.dim() == 4:
            logit = self.L[ti, typ, e.long(), pair_bins(game, A, a)]
        else:
            logit = self.L[ti, typ, e.long()]
        return logit, torch.zeros_like(a[:, 0])


class AllPairsPolicy(nn.Module):
    """Type-blind planners: 'random' (toggle each pair w.p. p) and 'maxconn' (add every absent tie)."""
    kind = 'fixed'

    def __init__(self, mode, p=0.30):
        super().__init__()
        self.mode, self.p = mode, p

    def forward(self, game, A, a, d, t):
        e, _ = pair_feats(game, A, a)
        if self.mode == 'random':
            q = torch.full_like(e, self.p)
        else:
            q = torch.where(e > 0.5, torch.zeros_like(e), torch.ones_like(e))
        q = q.clamp(1e-6, 1 - 1e-6)
        return torch.log(q) - torch.log1p(-q), torch.zeros_like(a[:, 0])


# ======================================================================== objectives
def step_reward(cfg, game, info):
    obj = cfg['obj']
    a, pay = game.a, game.pay
    if obj == 'e3':
        r = game.d.mean(-1)
    elif obj == 'welfare':
        r = pay.mean(-1)
    elif obj == 'omega':
        w = cfg.get('omega', 0.0)
        ben = game.ben * game.xn_now
        r = (ben * (a + w * (1 - a)) - game.cost * a * game.deg).mean(-1)
    elif obj == 'coop_mean':
        r = a.mean(-1)
    elif obj == 'mix':
        al = cfg.get('alpha', 0.5)
        r = al * a.mean(-1) + (1 - al) * pay.mean(-1) / S_W
    elif obj in ('coop_final', 'rawls', 'sustain'):
        r = torch.zeros_like(a[:, 0])
    else:
        raise ValueError(obj)
    P = cfg.get('P', 0.0)
    if P:
        r = r - P * info['rej'] / game.m
    if info['done']:
        if obj == 'coop_final':
            r = r + a.mean(-1)
        elif obj == 'rawls':
            r = r + game.d.min(-1).values
        elif obj == 'sustain':
            r = r + game.run_extra(cfg.get('K', 10))
    return r


# ======================================================================== PPO
class RunningNorm:
    def __init__(self):
        self.mean, self.var, self.n = 0.0, 1.0, 0

    def update(self, x):
        m, v = float(x.mean()), float(x.var())
        if self.n == 0:
            self.mean, self.var = m, max(v, 1e-6)
        else:
            self.mean = 0.95 * self.mean + 0.05 * m
            self.var = 0.95 * self.var + 0.05 * max(v, 1e-6)
        self.n += 1

    @property
    def std(self):
        return math.sqrt(self.var)


def rollout(pol, game, cfg, sample=True):
    obs = game.reset()
    S = dict(A=[], a=[], d=[], act=[], logp=[], v=[], r=[], rej=[], nrec=[])
    for t in range(1, NDEC + 1):
        A, a, d = obs['A'], obs['a'], obs['d']
        with torch.no_grad():
            logit, v = pol(game, A, a, d, t)
            p = torch.sigmoid(logit)
            act = torch.rand_like(p) < p if sample else p > 0.5
            logp = torch.where(act, torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit))
        info = game.step(act)
        r = step_reward(cfg, game, info)
        for k, x in zip(['A', 'a', 'd', 'act', 'logp', 'v', 'r', 'rej', 'nrec'],
                        [A, a, d, act, logp, v, r, info['rej'], info['nrec']]):
            S[k].append(x)
        obs = game.obs()
    return {k: torch.stack(v, 0) for k, v in S.items()}


def train(cfg, device='cuda', log_every=25):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    gen = torch.Generator(device=device)
    gen.manual_seed(1000 + cfg['seed'])
    B = cfg.get('batch', 2048)
    game = Game(B, device, gen=gen, **game_kw(cfg))
    pol = (GNNPolicy() if cfg['policy'] == 'gnn' else TablePolicy()).to(device)
    joint = cfg.get('ppo_mode', 'component') == 'joint'
    lr = cfg.get('lr', 3e-4 if cfg['policy'] == 'gnn' else 3e-3)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    iters = cfg.get('iters', 1000)
    gam, lam = cfg.get('gamma', 1.0), 0.95
    rn = RunningNorm()
    log = []
    t0 = time.time()
    for it in range(iters):
        frac = it / iters
        for g_ in opt.param_groups:
            g_['lr'] = lr * (1 - 0.9 * frac)
        ent_coef = cfg.get('ent', 0.01) * (1 - 0.9 * frac)
        S = rollout(pol, game, cfg)
        r, v = S['r'], S['v'] * rn.std + rn.mean      # value head predicts normalised return
        adv = torch.zeros_like(r)
        last = torch.zeros_like(r[0])
        for t in reversed(range(NDEC)):
            nv = v[t + 1] if t + 1 < NDEC else torch.zeros_like(v[0])
            delta = r[t] + gam * nv - v[t]
            last = delta + gam * lam * last
            adv[t] = last
        ret = adv + v
        rn.update(ret)
        ret_n = (ret - rn.mean) / rn.std
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx_all = torch.randperm(NDEC * B, device=device)
        nmb = cfg.get('minibatches', 4)
        flat = lambda x: x.reshape(NDEC * B, *x.shape[2:])
        FA, Fa, Fd = flat(S['A']), flat(S['a']), flat(S['d'])
        Fact, Flp, Fadv, Fret = flat(S['act']), flat(S['logp']), flat(adv), flat(ret_n)
        Ft = torch.arange(1, NDEC + 1, device=device).repeat_interleave(B)
        diag = dict(kl_joint=0.0, clip_comp=0.0, clip_joint=0.0, n=0)
        for ep in range(cfg.get('epochs', 4)):
            idx_all = torch.randperm(NDEC * B, device=device)
            for sel in idx_all.chunk(nmb):
                logit, vv = pol(game, FA[sel], Fa[sel], Fd[sel], Ft[sel])
                act = Fact[sel]
                lp = torch.where(act, torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit))
                dlp = lp - Flp[sel]
                ratio = torch.exp(dlp)
                with torch.no_grad():                       # diagnostics of the factorised update
                    jl = dlp.sum(-1)
                    diag['kl_joint'] += float(((torch.exp(jl) - 1) - jl).mean())   # k3 estimator of KL(old||new)
                    diag['clip_comp'] += float(((ratio - 1).abs() > 0.2).float().mean())
                    diag['clip_joint'] += float(((torch.exp(jl) - 1).abs() > 0.2).float().mean())
                    diag['n'] += 1
                if joint:                                   # standard PPO on the joint (product) action
                    jr = torch.exp(dlp.sum(-1).clamp(-20, 20))
                    A_ = Fadv[sel]
                    pg = -torch.min(jr * A_, jr.clamp(0.8, 1.2) * A_)
                else:                                       # per-component clipping, shared advantage
                    A_ = Fadv[sel].unsqueeze(-1)
                    pg = -torch.min(ratio * A_, ratio.clamp(0.8, 1.2) * A_).mean(-1)
                p = torch.sigmoid(logit)
                ent = -(p * torch.nn.functional.logsigmoid(logit) + (1 - p) * torch.nn.functional.logsigmoid(-logit)).mean(-1)
                vl = (vv - Fret[sel]) ** 2
                loss = (pg + 0.5 * vl - ent_coef * ent).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        if it % log_every == 0 or it == iters - 1:
            coop = torch.stack(game.hist['coop'], 0)
            row = dict(it=it, ret=float(S['r'].sum(0).mean()), coop_final=float(coop[-1].mean()),
                       coop_mean=float(coop.mean()), cap=float(game.d.mean()),
                       nrec=float(S['nrec'].mean()), sec=time.time() - t0,
                       ent=float(ent.detach().mean()), kl_joint=diag['kl_joint'] / max(diag['n'], 1),
                       clip_comp=diag['clip_comp'] / max(diag['n'], 1),
                       clip_joint=diag['clip_joint'] / max(diag['n'], 1))
            log.append(row)
            print(json.dumps(row), flush=True)
    return pol, log


# ======================================================================== evaluation
GAME_METRICS = ('coop_final', 'coop_ret', 'cap', 'rawls', 'welfare_ret', 'e3', 'desert', 'sustain', 'gap')


@torch.no_grad()
def evaluate(pol, kappa=1.0, n_games=20000, batch=4000, device='cuda', seed=12345, w_imit=0.0, K=10,
             cond=None, keep_games=False):
    """Monte Carlo evaluation. `cond` (dict of simulator keywords, see game_kw) overrides kappa/w_imit.
    With a fixed `seed`, every planner sees the same initial networks, dispositions and random-number
    stream (all draws have planner-independent shapes), so differences between planners evaluated with the
    same seed and condition are paired (common random numbers)."""
    cond = dict(cond) if cond else dict(kappa=kappa, w_imit=w_imit)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    keys = ['coop_final', 'coop_mean', 'cap', 'rawls', 'gini', 'gap', 'dens', 'sustain', 'e3', 'desert',
            'welfare_ret', 'coop_ret', 'nrec', 'rej', 'subsidy', 'benD_share', 'conv', 'degC', 'degD', 'capC',
            'capD', 'iso_final', 'coop_conn_final', 'forced', 'contrib', 'imit_flip', 'nchg', 'cap_p10']
    per = {k: [] for k in keys}
    traj = []
    rec = torch.zeros(NDEC, 3, 2, 2, device=device)       # [t, type, status, (opportunities, recs)]
    acc_stats = torch.zeros(NDEC, 3, 2, 2, device=device)  # [t, type, status, (recs, accepted changes)]
    done = 0
    while done < n_games:
        b = min(batch, n_games - done)
        game = Game(b, device, gen=gen, **game_kw(cond))
        obs = game.reset()
        rs_e3, rs_des, rejs, nrecs, nchg = [], [], [], [], []
        for t in range(1, NDEC + 1):
            A, a, d = obs['A'], obs['a'], obs['d']
            logit, _ = pol(game, A, a, d, t)
            act = torch.rand(logit.shape, device=device, generator=gen) < torch.sigmoid(logit)
            e, typ = pair_feats(game, A, a)
            e_before = e > 0.5
            info = game.step(act)
            e_after = game.pair_vals(game.A) > 0.5
            flipped = e_before ^ e_after
            for ty in range(3):
                for st in range(2):
                    msk = (typ == ty) & (e_before == bool(st))
                    rec[t - 1, ty, st, 0] += msk.float().sum()
                    rec[t - 1, ty, st, 1] += (msk & act).float().sum()
                    acc_stats[t - 1, ty, st, 0] += (msk & act).float().sum()
                    acc_stats[t - 1, ty, st, 1] += (msk & act & flipped).float().sum()
            rs_e3.append(game.d.mean(-1) - info['rej'] / game.m)
            ben = game.ben * game.xn_now
            rs_des.append((ben * game.a - game.cost * game.a * game.deg).mean(-1))
            rejs.append(info['rej'])
            nrecs.append(info['nrec'])
            nchg.append(flipped.float().sum(-1))
            obs = game.obs()
        h = {k: torch.stack(v, 0) for k, v in game.hist.items() if len(v)}
        disc = torch.tensor([0.99 ** i for i in range(NDEC)], device=device)
        fin_a = game.a
        dF = game.d.clone()
        nc = fin_a.sum(-1)
        per['coop_final'].append(h['coop'][-1])
        per['coop_mean'].append(h['coop'].mean(0))
        per['cap'].append(dF.mean(-1))
        per['rawls'].append(dF.min(-1).values)
        per['cap_p10'].append(torch.sort(dF, -1).values[:, :2].mean(-1))        # mean of the two poorest
        per['gini'].append(gini_t(dF))
        per['gap'].append((h['gap'] * h['mixed']).sum(0) / h['mixed'].sum(0).clamp(min=1))
        per['dens'].append(h['dens'][-1])
        per['e3'].append((torch.stack(rs_e3, 0) * disc[:, None]).sum(0))
        per['desert'].append(torch.stack(rs_des, 0).sum(0))
        per['welfare_ret'].append(h['pay'][1:].sum(0))
        per['coop_ret'].append(h['coop'][1:].sum(0))
        per['nrec'].append(torch.stack(nrecs, 0).mean(0))
        per['rej'].append(torch.stack(rejs, 0).mean(0))
        per['nchg'].append(torch.stack(nchg, 0).mean(0))
        num = h['nCD'][1:].sum(0)
        den = (2 * h['nCC'][1:] + h['nCD'][1:]).sum(0)
        per['subsidy'].append(torch.where(den > 0, num / den.clamp(min=1e-9), torch.zeros_like(num)))
        per['benD_share'].append(h['benD'][1:].sum(0) / h['ben'][1:].sum(0).clamp(min=1e-9))
        per['conv'].append(h['d2c'].sum(0) / h['nD'].sum(0).clamp(min=1))
        per['degC'].append(h['degC'][-1])
        per['degD'].append(h['degD'][-1])
        per['capC'].append((dF * fin_a).sum(-1) / nc.clamp(min=1))
        per['capD'].append((dF * (1 - fin_a)).sum(-1) / (game.n - nc).clamp(min=1))
        per['iso_final'].append(h['iso'][-1])
        per['coop_conn_final'].append(h['coop_conn'][-1])
        per['forced'].append(h['forced'][1:].mean(0))
        per['contrib'].append(h['contrib'][1:].mean(0))
        per['imit_flip'].append(h['imit_flip'][1:].mean(0))
        per['sustain'].append(game.run_extra(K))
        traj.append(h['coop'].mean(1) * b)
        done += b
    out = {}
    games = {}
    for k, v in per.items():
        x = torch.cat(v).double()
        out[k] = float(x.mean())
        out[k + '_se'] = float(x.std() / math.sqrt(len(x)))
        if keep_games and k in GAME_METRICS:
            games[k] = x.float().cpu().numpy()
    out['traj'] = (torch.stack(traj, 0).sum(0) / n_games).tolist()
    out['rec_rates'] = (rec[..., 1] / rec[..., 0].clamp(min=1)).tolist()
    out['rec_counts'] = rec.tolist()
    out['acc_rates'] = (acc_stats[..., 1] / acc_stats[..., 0].clamp(min=1)).tolist()
    out['chg_rates'] = (acc_stats[..., 1] / rec[..., 0].clamp(min=1)).tolist()   # realised changes per opportunity
    out['n_games'] = n_games
    out['cond'] = cond
    out['seed'] = seed
    out['kappa'] = cond.get('kappa', 1.0)
    out['w_imit'] = cond.get('w_imit', 0.0)
    if keep_games:
        out['_games'] = games
    return out


def table_probs(pol):
    return torch.sigmoid(pol.logit.detach()).cpu().numpy().tolist()


# ======================================================================== evolution strategies (table class)
class BatchTable(nn.Module):
    """Per-game logit tables L (B, 14, 3, 2) or state-augmented (B, 14, 3, 2, 3)."""
    kind = 'batchtable'

    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, game, A, a, d, t):
        e, typ = pair_feats(game, A, a)
        bi = torch.arange(a.shape[0], device=a.device)[:, None].expand_as(typ)
        ti = tidx(t, a)[:, None].expand_as(typ)
        if self.L.dim() == 5:
            return self.L[bi, ti, typ, e.long(), pair_bins(game, A, a)], torch.zeros_like(a[:, 0])
        return self.L[bi, ti, typ, e.long()], torch.zeros_like(a[:, 0])


def logit_table(P, lo=0.03, hi=0.97):
    P = np.clip(np.asarray(P, float), lo, hi)
    return np.log(P) - np.log1p(-P)


FREEZE_SETS = {
    # distilled encouragement schedule imposed on C-D pairs in rounds 8-14 (late exclusion)
    'late_cd': [(t, 1, st) for t in range(7, 14) for st in (0, 1)],
}


def episode_return(pol, game, cfg, A0=None, th0=None):
    obs = game.reset(A0, th0)
    gam = cfg.get('gamma', 1.0)
    R = torch.zeros(game.B, device=game.dev)
    for t in range(1, NDEC + 1):
        with torch.no_grad():
            logit, _ = pol(game, obs['A'], obs['a'], obs['d'], t)
            act = torch.rand(logit.shape, device=game.dev, generator=game.g) < torch.sigmoid(logit)
        info = game.step(act)
        R = R + (gam ** (t - 1)) * step_reward(cfg, game, info)
        obs = game.obs()
    return R


def train_es(cfg, device='cuda', log_every=10):
    """OpenAI-style ES with antithetic sampling, common initial conditions per antithetic pair,
    centred-rank fitness shaping and Adam."""
    torch.manual_seed(cfg['seed'])
    gen = torch.Generator(device=device)
    gen.manual_seed(1000 + cfg['seed'])
    npair, G = cfg.get('npair', 128), cfg.get('games', 32)
    sigma, lr, iters = cfg.get('sigma', 0.5), cfg.get('lr', 0.1), cfg.get('iters', 200)
    B = 2 * npair * G
    game = Game(B, device, gen=gen, **game_kw(cfg))
    half = Game(npair * G, device, gen=gen, **{k: v for k, v in game_kw(cfg).items() if k == 'mu_th'})
    shape = (NDEC, 3, 2, 3) if cfg['policy'] == 'table_r' else (NDEC, 3, 2)
    theta = torch.zeros(*shape, device=device)
    init = cfg.get('init')
    if init:
        P0 = fixed_tables()[init]
        L0 = torch.as_tensor(logit_table(P0), dtype=torch.float32, device=device)
        theta = L0[..., None].expand(*shape).clone() if len(shape) == 4 else L0.clone()
    mask = torch.ones_like(theta)
    if cfg.get('freeze'):
        Lenc = torch.as_tensor(logit_table(enc_table(), 1e-3, 1 - 1e-3), dtype=torch.float32, device=device)
        for (t_, ty, st) in FREEZE_SETS[cfg['freeze']]:
            mask[t_, ty, st] = 0.0
            theta[t_, ty, st] = Lenc[t_, ty, st]
    theta0 = theta.clone()
    m_, v_ = torch.zeros_like(theta), torch.zeros_like(theta)
    b1, b2 = 0.9, 0.999
    log = []
    t0 = time.time()
    for it in range(iters):
        eps = torch.randn(npair, *shape, device=device, generator=gen) * mask
        L = torch.cat([theta + sigma * eps, theta - sigma * eps], 0)            # (2*npair, ...)
        Lg = L.repeat_interleave(G, 0)
        half.reset()
        A0 = torch.cat([half.A, half.A], 0)
        th0 = torch.cat([half.theta, half.theta], 0)
        R = episode_return(BatchTable(Lg), game, cfg, A0, th0)
        f = R.view(2 * npair, G).mean(1)
        rk = torch.empty_like(f)
        rk[f.argsort()] = torch.arange(len(f), device=device, dtype=f.dtype)
        rk = rk / (len(f) - 1) - 0.5
        w = rk[:npair] - rk[npair:]
        grad = -(w.view(-1, *([1] * len(shape))) * eps).sum(0) / (npair * sigma)
        lr_t = lr * (1 - 0.8 * it / iters)
        m_ = b1 * m_ + (1 - b1) * grad
        v_ = b2 * v_ + (1 - b2) * grad ** 2
        mh, vh = m_ / (1 - b1 ** (it + 1)), v_ / (1 - b2 ** (it + 1))
        theta = (theta - lr_t * mask * mh / (vh.sqrt() + 1e-8)).clamp(-8, 8)
        theta = mask * theta + (1 - mask) * theta0
        if it % log_every == 0 or it == iters - 1:
            coop = torch.stack(game.hist['coop'], 0)
            row = dict(it=it, ret=float(f.mean()), coop_final=float(coop[-1].mean()), cap=float(game.d.mean()),
                       sec=time.time() - t0)
            log.append(row)
            print(json.dumps(row), flush=True)
    pol = TablePolicyR() if cfg['policy'] == 'table_r' else TablePolicy()
    pol.logit.data = theta.cpu()
    return pol.to(device), log
