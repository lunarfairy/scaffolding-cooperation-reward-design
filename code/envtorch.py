"""Batched (GPU) re-implementation of the McKee et al. (2023) cooperative network game.

Game rules and the 'bot' model of human players follow the Supplementary Information
(Sections A, B, E4; Supp. Tables 1 and 3), with the beta-label reading consistent with SI D3:
  round 1 logit      = 1.807 + 0.818*theta
  later rounds logit = -0.010 - 0.193*deg + k*(0.370*xn + 1.521*xr) + theta,
where k = 1 for last-round cooperators and k = kappa for last-round defectors (kappa = 1 is the
fitted model). Recommendation acceptance phi depends on the valence and the referent's action.

The two parameter blocks of Supp. Table 3 are labelled (beta_0, beta_1) and (beta'_0..beta'_3), whereas the
Methods equation uses two primed parameters for round 1 and four unprimed parameters for later rounds.
Only one assignment is consistent with the parameter counts (the two-parameter block must be the round-1
model), so the reading above is forced up to the label swap.

Optional perturbations (all default to the published simulator):
  kappa   multiplier on the neighbourhood terms (xn, xr) for last-round defectors
  delta   additive intercept shift for last-round defectors (used for propensity-matched variants)
  ben/cost game parameters b and c;  mu_th: mean of the disposition distribution
  w_imit, temp, imit: with probability w_imit a player imitates one random neighbour, copying its
          previous action with Fermi probability sigmoid((pi_j - pi_i)/temp) ('payoff') or with
          probability 1/2 regardless of payoffs ('unbiased').
Order within a round: logistic choice -> (optional) imitation override -> capital gate (SI eq. 2).
Isolated players (degree 0) have x_n = x_r = 0, cannot imitate, and their choice is recorded as
cooperation even though it yields no payoff; the isolation rate is recorded separately.
"""
import torch

BEN, COST, D0 = 0.10, 0.05, 1.0
N_PLAYERS, T_ROUNDS, P_ER = 16, 15, 0.3
MU_TH, SD_TH = -0.304, 2.410
R1 = (1.807, 0.818)
RL = (-0.010, -0.193, 0.370, 1.521)
PHI_DEL_D, PHI_DEL_C, PHI_ADD_D, PHI_ADD_C = 0.774, 0.085, 0.287, 0.909


class Game:
    """B independent games. Planner acts after rounds t = 1..T-1 (14 decisions)."""

    def __init__(self, B, device='cpu', kappa=1.0, w_imit=0.0, temp=0.1, n=N_PLAYERS, T=T_ROUNDS,
                 p=P_ER, gen=None, delta=0.0, ben=BEN, cost=COST, mu_th=MU_TH, imit='payoff'):
        self.B, self.n, self.T, self.p = B, n, T, p
        self.dev = torch.device(device)
        self.kappa, self.w_imit, self.temp = kappa, w_imit, temp
        self.delta, self.ben, self.cost, self.mu_th, self.imit = delta, ben, cost, mu_th, imit
        self.g = gen
        iu = torch.triu_indices(n, n, 1, device=self.dev)
        self.iu, self.ju = iu[0], iu[1]
        self.m = self.iu.numel()

    # ---------------------------------------------------------------- utils
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.g)

    def pair_vals(self, M):
        return M[:, self.iu, self.ju]

    def set_pairs(self, v):
        M = torch.zeros(self.B, self.n, self.n, dtype=v.dtype, device=self.dev)
        M[:, self.iu, self.ju] = v
        return M + M.transpose(1, 2)

    # ---------------------------------------------------------------- dynamics
    def reset(self, A0=None, theta0=None):
        B, n = self.B, self.n
        if A0 is None:
            e = self._rand(B, self.m) < self.p
            self.A = self.set_pairs(e.float())
            self.theta = self.mu_th + SD_TH * torch.randn(B, n, device=self.dev, generator=self.g)
        else:
            self.A, self.theta = A0.clone(), theta0.clone()
        self.d = torch.full((B, n), D0, device=self.dev)
        self.t = 1
        self.a_prev = None
        self.pay_prev = torch.zeros(B, n, device=self.dev)
        self.hist = dict(coop=[], pay=[], nCC=[], nCD=[], nDD=[], gap=[], mixed=[], dens=[],
                         benD=[], ben=[], degC=[], degD=[], d2c=[], nD=[], iso=[], coop_conn=[],
                         forced=[], contrib=[], imit_flip=[])
        self._play()
        return self.obs()

    def _play(self, record=True):
        A, n = self.A, self.n
        deg = A.sum(-1)
        if self.a_prev is None:
            logit = R1[0] + R1[1] * self.theta
        else:
            ap = self.a_prev
            xn = torch.bmm(A, ap.unsqueeze(-1)).squeeze(-1)
            xr = xn / deg.clamp(min=1.0)
            k = torch.where(ap > 0.5, torch.ones_like(ap), torch.full_like(ap, self.kappa))
            logit = RL[0] + RL[1] * deg + k * (RL[2] * xn + RL[3] * xr) + self.theta
            if self.delta:
                logit = logit + self.delta * (ap < 0.5).float()
        a = (self._rand(self.B, n) < torch.sigmoid(logit)).float()
        a_logit = a
        if self.a_prev is not None and self.w_imit > 0:
            # payoff-biased (Fermi) imitation of one random neighbour, used with prob w_imit
            use = self._rand(self.B, n) < self.w_imit
            gum = -torch.log(-torch.log(self._rand(self.B, n, n).clamp(1e-9, 1 - 1e-9)))
            gum = torch.where(A > 0.5, gum, torch.full_like(gum, -1e9))
            j = gum.argmax(-1)
            has = deg > 0
            pj = torch.gather(self.pay_prev, 1, j)
            aj = torch.gather(self.a_prev, 1, j)
            cp = torch.sigmoid((pj - self.pay_prev) / self.temp)
            if self.imit == 'unbiased':
                cp = torch.full_like(cp, 0.5)
            copy = self._rand(self.B, n) < cp
            a_im = torch.where(copy, aj, self.a_prev)
            a = torch.where(use & has, a_im, a)
        a_pre = a
        a = a * (self.cost * deg <= self.d + 1e-6).float()     # capital constraint (SI eq. 2)
        xn_now = torch.bmm(A, a.unsqueeze(-1)).squeeze(-1)
        pay = self.ben * xn_now - self.cost * a * deg
        self.d = self.d + pay
        if record:
            h = self.hist
            h['coop'].append(a.mean(-1))
            h['pay'].append(pay.mean(-1))
            s = self.pair_vals(a[:, :, None] + a[:, None, :])
            e = self.pair_vals(A)
            h['nCC'].append(((s == 2).float() * e).sum(-1))
            h['nCD'].append(((s == 1).float() * e).sum(-1))
            h['nDD'].append(((s == 0).float() * e).sum(-1))
            h['dens'].append(e.mean(-1))
            nc = a.sum(-1)
            mixed = (nc > 0) & (nc < n)
            pc_ = (pay * a).sum(-1) / nc.clamp(min=1)
            pd_ = (pay * (1 - a)).sum(-1) / (n - nc).clamp(min=1)
            h['gap'].append(torch.where(mixed, pd_ - pc_, torch.zeros_like(pc_)))
            h['mixed'].append(mixed.float())
            h['benD'].append((self.ben * xn_now * (1 - a)).mean(-1))  # benefits received by defectors
            h['ben'].append((self.ben * xn_now).mean(-1))
            conn = (deg > 0).float()
            h['iso'].append(1 - conn.mean(-1))
            h['coop_conn'].append((a * conn).sum(-1) / conn.sum(-1).clamp(min=1))
            h['forced'].append((a_pre - a).mean(-1))                  # intended C blocked by capital gate
            h['contrib'].append((a * deg).mean(-1))                     # cooperative ties paid for, per player
            h['imit_flip'].append((a_pre != a_logit).float().mean(-1))
            h['degC'].append((deg * a).sum(-1) / nc.clamp(min=1))
            h['degD'].append((deg * (1 - a)).sum(-1) / (n - nc).clamp(min=1))
            if self.a_prev is not None:                               # D -> C conversions
                wasD = 1 - self.a_prev
                h['d2c'].append((wasD * a).sum(-1))
                h['nD'].append(wasD.sum(-1))
        self.a, self.pay, self.deg, self.xn_now = a, pay, deg, xn_now
        self.a_prev, self.pay_prev = a, pay

    def load_state(self, A, theta, a, pay, d, t):
        """Set the game to the state reached after round t has been played (used for counterfactuals)."""
        self.A, self.theta = A.clone(), theta.clone()
        self.a, self.pay, self.d = a.clone(), pay.clone(), d.clone()
        self.a_prev, self.pay_prev = self.a, self.pay
        self.deg = A.sum(-1)
        self.t = t
        self.hist = {k: [] for k in self.hist} if hasattr(self, 'hist') else dict(
            coop=[], pay=[], nCC=[], nCD=[], nDD=[], gap=[], mixed=[], dens=[], benD=[], ben=[], degC=[],
            degD=[], d2c=[], nD=[], iso=[], coop_conn=[], forced=[], contrib=[], imit_flip=[])

    def obs(self):
        return dict(A=self.A, a=self.a, d=self.d, t=self.t)

    def step(self, R):
        """R: (B, m) bool, True = recommend a change on pair (add if absent, delete if present).
        The recommendation goes to a random endpoint; the other endpoint is the referent."""
        R = R.bool()
        e = self.pair_vals(self.A) > 0.5
        ai, aj = self.a[:, self.iu], self.a[:, self.ju]
        u = self._rand(self.B, self.m) < 0.5
        a_ref = torch.where(u, ai, aj) > 0.5
        phi = torch.where(e, torch.where(a_ref, torch.full_like(ai, PHI_DEL_C), torch.full_like(ai, PHI_DEL_D)),
                          torch.where(a_ref, torch.full_like(ai, PHI_ADD_C), torch.full_like(ai, PHI_ADD_D)))
        acc = self._rand(self.B, self.m) < phi
        flip = R & acc
        rej = (R & ~acc).float().sum(-1)
        nrec = R.float().sum(-1)
        self.A = self.set_pairs((e ^ flip).float())
        self.t += 1
        self._play()
        return dict(rej=rej, nrec=nrec, done=self.t >= self.T)

    def run_extra(self, K):
        """Planner withdraws; network frozen; K further rounds. Returns mean cooperation (B,)."""
        cs = []
        for _ in range(K):
            self._play(record=False)
            cs.append(self.a.mean(-1))
        return torch.stack(cs, 0).mean(0)


def gini_t(x):
    x, _ = torch.sort(x, dim=-1)
    n = x.shape[-1]
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    s = x.sum(-1)
    g = ((2 * idx - n - 1) * x).sum(-1) / (n * s.clamp(min=1e-9))
    return torch.where(s > 0, g, torch.zeros_like(g))
