"""Manuscript figures (revision). Inputs are the tidy outputs of eval_all.py / paired.py / marginal.py /
coefprop.py / calib.py; see make_figures.py for the entry point."""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import analysis2 as A
from figstyle import COL, LABEL, SHORT, CORE, apply_style, panel_letter

KG = [0.5, 0.625, 0.75, 0.875, 1.0]
MAIN5 = ['e3', 'welfare', 'desert', 'coopmean', 'coopfinal']
KKEY = {0.5: 'k0.5', 0.625: 'k0.625', 0.75: 'k0.75', 0.875: 'k0.875', 1.0: 'k1'}


def own(E, cond, kind, fam, seed=12345):
    return E[(E.eval_cond == cond) & (E.seed == seed) & E.own & (E.kind == kind) & (E.family == fam)]


def rate_by_round(sub, pair, status, which='rec_rates'):
    num = np.zeros(14); den = np.zeros(14)
    for _, r in sub.iterrows():
        opp = np.array(r['rec_opp'])[:, pair, status]
        num += np.array(r[which])[:, pair, status] * opp
        den += opp
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def phase(sub, pair, status, rounds, which='rec_rates'):
    num = den = 0.0
    for _, r in sub.iterrows():
        opp = np.array(r['rec_opp'])[:, pair, status][rounds]
        num += (np.array(r[which])[:, pair, status][rounds] * opp).sum(); den += opp.sum()
    return num / den if den else np.nan


# ----------------------------------------------------------------------------------------- Figure 1
def fig1(E, coef, marg, calmae, path):
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15), gridspec_kw=dict(wspace=0.62))
    # a: omega threshold, kappa = 0.5, b = 0.10 vs 0.15
    ax = axs[0]
    fams = [('desert', 0.0), ('omega0.25', 0.25), ('omega0.5', 0.5), ('omega0.75', 0.75), ('welfare', 1.0)]
    for cond, bval, col in [('k0.5', 0.10, '#542788'), ('k0.5_ben0.15', 0.15, '#e08214')]:
        xs, ys, lo, hi = [], [], [], []
        for f, w in fams:
            sub = own(E, cond, 'table', f)
            if not len(sub):
                continue
            per = [phase(sub.iloc[[i]], 1, 1, list(range(7, 14))) for i in range(len(sub))]
            xs.append(w); ys.append(np.mean(per)); lo.append(min(per)); hi.append(max(per))
        ax.plot(xs, ys, 'o-', color=col, label=f'b = {bval:.2f}')
        ax.vlines(xs, lo, hi, color=col, lw=0.6)
        ax.axvline(0.05 / bval, color=col, ls=':', lw=0.8)
    ax.set_xlabel('Weight on defectors\' benefits, ω')
    ax.set_ylabel('C–D cut rate, rounds 8–14')
    ax.set_title('Exclusion ends near ω* = c/b')
    ax.legend(loc='lower left', handlelength=1.2)
    ax.set_ylim(-0.03, 1.03)
    panel_letter(ax, 'a')
    # b: one-step value vs kappa
    ax = axs[1]
    pt = coef[(coef.draw == 'point') & (coef.spec == 'base')]
    kk = np.round(np.arange(0, 1.5001, 0.05), 3)
    for (t, g), col in zip(pt.groupby('t'), ['#bdbdbd', '#737373', '#252525']):
        cv = np.array(json.loads(g.curve.iloc[0]))
        ax.plot(kk, cv * 100, color=col, label=f'after round {t}')
    dr = coef[(coef.draw != 'point') & (coef.t == 7)].break_even.astype(float)
    q = np.nanpercentile(dr, [2.5, 50, 97.5])
    ax.errorbar([q[1]], [0], xerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt='D', color='#d6604d', ms=3, lw=0.8, capsize=2)
    ax.text(q[1], -0.35, 'break-even κ\n(95% coef. range)', color='#d6604d', ha='center', va='top', fontsize=6)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel('Defector responsiveness, κ')
    ax.set_ylabel('Δ cooperators next round (×10$^{-2}$)')
    ax.set_title('Tie value changes sign with κ')
    ax.legend(loc='upper left', handlelength=1.2)
    panel_letter(ax, 'b')
    # c: decomposition (encouragement states, round 7)
    ax = axs[2]
    parts = [('dC_deg', 'C: degree'), ('dC_nb', 'C: neighbours'), ('dD_deg', 'D: degree'), ('dD_nb', 'D: neighbours')]
    w = 0.38
    for j, (kap, col) in enumerate([(1.0, '#4d4d4d'), (0.5, '#b2abd2')]):
        r = marg[(marg.planner == 'encouragement') & (marg.kappa == kap) & (marg.t == 7)].iloc[0]
        vals = [r[p] * 100 for p, _ in parts] + [r['dTot'] * 100]
        x = np.arange(len(vals)) + (j - 0.5) * w
        ax.bar(x, vals, w, color=col, label=f'κ = {kap:g}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([p for _, p in parts] + ['Total'], rotation=45, ha='right')
    ax.set_ylabel('Δ cooperation prob. (×10$^{-2}$)')
    ax.set_title('Fixed degree cost, scaled gain')
    ax.legend(loc='upper left', handlelength=1)
    panel_letter(ax, 'c')
    # d: calibration
    ax = axs[3]
    base = calmae[calmae.kind == 'kappa'].sort_values('kappa')
    ax.plot(base.kappa, base.mae_final, 'o-', color='#2166ac', label='κ grid')
    for _, r in calmae[calmae.kind != 'kappa'].iterrows():
        ax.plot([r.kappa], [r.mae_final], r.marker, color=r.color, mfc='white' if r.marker == 's' else r.color, ms=4,
                label=r.label)
    ax.set_xlabel('Defector responsiveness, κ')
    ax.set_ylabel('Mean abs. error, final coop.')
    ax.set_title('Fit to human outcomes')
    ax.set_ylim(0, None)
    ax.legend(loc='upper left', handlelength=1, fontsize=6)
    panel_letter(ax, 'd')
    fig.savefig(path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------------------- Figure 2
def fig2(E, path, kinds=('table', 'gnn')):
    """Opportunity-weighted C-D recommendation rates by round (schedule ES, solid; graph network, dashed)."""
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.25))
    t = np.arange(1, 15)
    for i, kap in enumerate([1.0, 0.5]):
        cond = KKEY[kap]
        for j, (st, nm) in enumerate([(0, 'Add rate, unconnected C–D pairs'), (1, 'Cut rate, connected C–D pairs')]):
            ax = axs[i, j]
            for f in MAIN5:
                for kind, ls in zip(kinds, ['-', '--']):
                    sub = own(E, cond, kind, f)
                    if len(sub):
                        ax.plot(t, rate_by_round(sub, 1, st), ls, color=COL[f], lw=1.1 if kind == 'table' else 0.8,
                                label=SHORT[f] if (kind == 'table') else None)
            sub = own(E, cond, 'fixed', 'encouragement')
            ax.plot(t, rate_by_round(sub, 1, st), ':', color='k', lw=1.3, label='Distilled published' if True else None)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(nm)
            if j == 0:
                ax.set_ylabel(f'κ = {kap:g}\nrecommendation rate')
            if i == 1:
                ax.set_xlabel('Round')
            panel_letter(ax, 'abcd'[2 * i + j], dx=-0.12)
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], color='0.3', ls='-'), Line2D([], [], color='0.3', ls='--')]
    l += ['schedule (ES)', 'graph network (PPO)']
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), handlelength=1.6)
    fig.savefig(path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------------------- Figure 3
def seedstats(E, cond, kind, fam, m):
    sub = own(E, cond, kind, fam)
    x = sub[m].astype(float).values
    if len(x) == 0:
        return np.nan, np.nan
    ci = 0 if len(x) < 2 else 2.776 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) == 5 else \
        {2: 12.706, 3: 4.303, 4: 3.182}.get(len(x), 2.0) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), ci


def fig3(E, path):
    apply_style()
    mets = [('coop_final', 'Final-round cooperation', 'Cooperation diverges as κ falls'),
            ('cap', 'Mean final capital', 'Payoff rewards keep capital'),
            ('rawls', 'Minimum final capital', 'Exclusion costs the poorest'),
            ('iso_final', 'Isolated players, final round', 'Cooperation rewards isolate')]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.5))
    for ax, (m, yl, tt), L in zip(axs, mets, 'abcd'):
        for f in CORE:
            mu = [seedstats(E, KKEY[k], 'table', f, m) for k in KG]
            ax.errorbar(KG, [a for a, _ in mu], yerr=[b for _, b in mu], color=COL[f], marker='o', ms=2.5, lw=1,
                        capsize=0, label=SHORT[f])
            gm, glo, ghi = [], [], []
            for k in [0.5, 0.75, 1.0]:
                x = own(E, KKEY[k], 'gnn', f)[m].astype(float).values
                gm.append(x.mean() if len(x) else np.nan)
                glo.append(x.mean() - x.min() if len(x) else 0); ghi.append(x.max() - x.mean() if len(x) else 0)
            ax.errorbar(np.array([0.5, 0.75, 1.0]) + 0.012, gm, yerr=[glo, ghi], fmt='s',
                        color=COL[f], mfc='white', ms=2.8, lw=0.6, capsize=0)
        for f, ls in [('encouragement', ':'), ('conciliation', '--'), ('exclusion', '-.')]:
            v = [own(E, KKEY[k], 'fixed', f)[m].mean() for k in KG]
            ax.plot(KG, v, ls, color=COL[f], lw=1, label=SHORT[f])
        ax.set_xlabel('Defector responsiveness, κ')
        ax.set_ylabel(yl)
        ax.set_title(tt)
        ax.set_xticks([0.5, 0.75, 1.0])
        panel_letter(ax, L)
    h, l = axs[0].get_legend_handles_labels()
    h += [Line2D([], [], color='0.3', marker='o', ls='-', ms=2.5), Line2D([], [], color='0.3', marker='s', mfc='white', ls='', ms=3)]
    l += ['schedule (ES)', 'graph network (PPO)']
    fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.2), handlelength=1.6)
    fig.savefig(path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------------------- Figure 4
def fig4(E, path):
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.25), gridspec_kw=dict(wspace=0.55))
    core5 = ['e3', 'welfare', 'desert', 'coopmean', 'coopfinal']
    # a: frozen transfer to kappa = 0.5 as a function of training kappa
    ax = axs[0]
    for f in core5:
        sub = E[(E.seed == 12345) & (E.kind == 'table') & (E.family == f) & (E.eval_cond == 'k0.5')]
        ys = [sub[sub.train_cond == KKEY[k]].coop_final.mean() for k in KG]
        ax.plot(KG, ys, 'o-', color=COL[f], ms=2.5, label=SHORT[f])
        g = E[(E.seed == 12345) & (E.kind == 'gnn') & (E.family == f) & (E.eval_cond == 'k0.5')]
        gy = [g[g.train_cond == KKEY[k]].coop_final.mean() for k in [0.5, 0.75, 1.0]]
        ax.plot(np.array([0.5, 0.75, 1.0]) + 0.012, gy, 's', color=COL[f], mfc='white', ms=2.8)
    ax.set_xlabel('Training κ (tested at κ = 0.5)')
    ax.set_ylabel('Final-round cooperation')
    ax.set_title('Frozen planners transferred')
    ax.set_xticks([0.5, 0.75, 1.0])
    panel_letter(ax, 'a')
    # b: slope versus level (top: final cooperation; bottom: late C-D cut rate)
    ax.figure.delaxes(axs[1])
    sp = axs[1].get_subplotspec().subgridspec(2, 1, hspace=0.12, height_ratios=[1.4, 1])
    axt, axb = fig.add_subplot(sp[0]), fig.add_subplot(sp[1])
    conds = [('k1', 'fitted'), ('k0.5_delta2.089', 'slope'), ('k1_delta-2.209', 'level'), ('k0.5', 'both')]
    for j, (c, lab) in enumerate(conds):
        for i, f in enumerate(core5):
            sub = own(E, c, 'table', f)
            x = j + (i - 2) * 0.13
            axt.plot(x, sub.coop_final.mean(), 'o', color=COL[f], ms=2.8)
            axb.bar(x, phase(sub, 1, 1, list(range(7, 14))), 0.12, color=COL[f])
    axt.set_xticks(range(4)); axt.set_xticklabels([])
    axb.set_xticks(range(4)); axb.set_xticklabels([l for _, l in conds], fontsize=6)
    axt.set_ylabel('Final coop.'); axb.set_ylabel('Late cut rate')
    axb.set_ylim(0, 1.02)
    axt.set_title('Slope versus propensity')
    axb.set_xlabel('Reduced defector term')
    panel_letter(axt, 'b')
    # c: imitation strength
    ax = axs[2]
    ws = [0.0, 0.1, 0.3, 0.5]
    ck = {0.0: 'k1', 0.1: 'k1_w_imit0.1', 0.3: 'k1_w_imit0.3', 0.5: 'k1_w_imit0.5'}
    for f in core5:
        ax.plot(ws, [own(E, ck[w], 'table', f).coop_final.mean() for w in ws], 'o-', color=COL[f], ms=2.5)
        ax.plot([0.3 + 0.02], [own(E, 'k1_w_imit0.3_imitunbiased', 'table', f).coop_final.mean()], 'D', color=COL[f],
                mfc='white', ms=3)
    ax.plot(ws, [own(E, ck[w], 'fixed', 'encouragement').coop_final.mean() for w in ws], ':', color='k')
    ax.set_xlabel('Imitation probability, w')
    ax.set_ylabel('Final-round cooperation')
    ax.set_title('Imitation widens the gap')
    ax.plot([], [], 'D', color='0.3', mfc='white', ms=3, label='payoff-unbiased copy')
    ax.legend(loc='lower left', fontsize=6, handlelength=1)
    panel_letter(ax, 'c')
    # d: reward accounting factorial
    ax = axs[3]
    cells = [('e3P0', 'Lev. P0'), ('e3', 'Lev. P1'), ('welfare', 'Inc. P0'), ('welfarePm', 'Inc. Pm'),
             ('coopmeanPm', 'Coop Pm'), ('coopmean', 'Coop P0')]
    for j, (c, col, lab) in enumerate([('k0.5', '#542788', 'κ = 0.5'), ('k1_w_imit0.3', '#1b7837', 'κ = 1, w = 0.3')]):
        ys = []
        for i, (f, _) in enumerate(cells):
            sub = own(E, c, 'table', f)
            ys.append(sub.coop_final.mean())
            ax.plot([i + (j - 0.5) * 0.25] * len(sub), sub.coop_final, '.', color=col, ms=2, alpha=0.6)
        ax.plot(np.arange(len(cells)) + (j - 0.5) * 0.25, ys, '_', color=col, ms=8, mew=1.5, label=lab)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([l for _, l in cells], rotation=45, ha='right', fontsize=6)
    ax.set_xlabel('Reward accounting, penalty')
    ax.set_ylabel('Final-round cooperation')
    ax.set_title('What is valued matters most')
    ax.legend(loc='lower right', fontsize=6, handlelength=1)
    panel_letter(ax, 'd')
    h = [Line2D([], [], color=COL[f], marker='o', ms=2.5) for f in core5] + \
        [Line2D([], [], color='0.3', marker='s', mfc='white', ls='', ms=3), Line2D([], [], color='k', ls=':')]
    l = [SHORT[f] for f in core5] + ['graph network (PPO)', 'Distilled published']
    fig.legend(h, l, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.3), handlelength=1.6)
    fig.savefig(path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------------------- Figure 5
ROWS5 = [('table', 'e3'), ('table', 'welfare'), ('table', 'desert'), ('table', 'rawls'), ('table', 'coopmean'),
         ('table', 'coopfinal'), ('table', 'sustain'), ('gnn', 'e3'), ('gnn', 'welfare'), ('gnn', 'coopfinal'),
         ('fixed', 'encouragement'), ('fixed', 'conciliation'), ('fixed', 'exclusion')]
COLS5 = ['e3', 'welfare', 'desert', 'rawls', 'coopmean', 'coopfinal', 'sustain']


def fig5(P, E, path):
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 0.8, 0.8], wspace=0.5)
    for k, (cond, tt) in enumerate([('k1', 'Fitted model, κ = 1'), ('k0.5', 'κ = 0.5 (best fit to humans)')]):
        ax = fig.add_subplot(gs[k])
        M = P[P.cond == cond].pivot_table(index=['kind', 'family'], columns='obj', values='regret')
        rows = [r for r in ROWS5 if r in M.index]
        Z = M.loc[rows, COLS5].values
        ax.imshow(np.clip(Z, 0, 1), cmap='Greys', vmin=0, vmax=1, aspect='auto')
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                ax.text(j, i, f'{v:.2f}' if v < 0.995 else f'{v:.1f}', ha='center', va='center', fontsize=5,
                        color='white' if v > 0.55 else 'black')
        ax.set_xticks(range(len(COLS5))); ax.set_xticklabels([SHORT[c] for c in COLS5], rotation=60, ha='right', fontsize=6)
        lab = [('' if kd == 'fixed' else ('ES: ' if kd == 'table' else 'GNN: ')) + SHORT[f] for kd, f in rows]
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(lab if k == 0 else [], fontsize=6)
        ax.set_title(tt)
        if k == 0:
            ax.set_xlabel('Objective evaluated')
        panel_letter(ax, 'ab'[k], dx=-0.55 if k == 0 else -0.1)
    # c: cost of imposing late exclusion under payoff rewards
    ax = fig.add_subplot(gs[2])
    for i, f in enumerate(['e3', 'welfare']):
        for j, kap in enumerate([0.5, 0.75, 1.0]):
            c = KKEY[kap]
            base = own(E, c, 'table', f)['J_' + f]
            con = own(E, c, 'table', f + '-latecd')['J_' + f]
            d = 100 * (con.values - base.mean()) / base.mean()
            x = j + (i - 0.5) * 0.3
            ax.plot([x] * len(d), d, '.', color=COL[f], ms=3)
            ax.plot([x], [d.mean()], '_', color=COL[f], ms=8, mew=1.5)
            sd = 100 * base.std(ddof=1) / base.mean()
            ax.fill_between([x - 0.12, x + 0.12], -2 * sd, 2 * sd, color=COL[f], alpha=0.25, lw=0)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['0.5', '0.75', '1'])
    ax.set_xlabel('κ')
    ax.set_ylabel('Change in own objective (%)')
    ax.set_title('Imposed late exclusion')
    ax.plot([], [], 's', color=COL['e3'], label=SHORT['e3']); ax.plot([], [], 's', color=COL['welfare'], label=SHORT['welfare'])
    ax.legend(loc='lower left', fontsize=6, handlelength=1)
    panel_letter(ax, 'c')
    # d: distilled schedule regret profile across kappa
    ax = fig.add_subplot(gs[3])
    for cond, col, lab in [('k1', '#252525', 'κ = 1'), ('k0.75', '#737373', 'κ = 0.75'), ('k0.5', '#bdbdbd', 'κ = 0.5')]:
        sub = P[(P.cond == cond) & (P.kind == 'fixed') & (P.family == 'encouragement')].set_index('obj')
        if len(sub):
            ax.plot(range(len(COLS5)), sub.loc[COLS5, 'regret'], 'o-', color=col, ms=2.5, label=lab)
    ax.set_xticks(range(len(COLS5))); ax.set_xticklabels([SHORT[c] for c in COLS5], rotation=60, ha='right', fontsize=6)
    ax.set_ylabel('Normalized regret')
    ax.set_title('Distilled schedule')
    ax.set_ylim(0, None)
    ax.legend(loc='upper left', fontsize=6, handlelength=1)
    panel_letter(ax, 'd')
    fig.savefig(path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------------------- Extended Data
def ed_sched(E, path):
    """All pair types: C-C add, D-D cut, realised C-D additions and cuts per opportunity (ES, 7 objectives)."""
    apply_style()
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.35))
    t = np.arange(1, 15)
    spec = [(2, 0, 'rec_rates', 'C–C add rate'), (0, 1, 'rec_rates', 'D–D cut rate'),
            (1, 0, 'chg_rates', 'C–D ties formed / opportunity'), (1, 1, 'chg_rates', 'C–D ties cut / opportunity')]
    for i, kap in enumerate([1.0, 0.5]):
        for j, (pair, st, which, tt) in enumerate(spec):
            ax = axs[i, j]
            for f in CORE:
                sub = own(E, KKEY[kap], 'table', f)
                ax.plot(t, rate_by_round(sub, pair, st, which), color=COL[f], lw=0.9, label=SHORT[f])
            sub = own(E, KKEY[kap], 'fixed', 'encouragement')
            ax.plot(t, rate_by_round(sub, pair, st, which), ':', color='k', lw=1.2, label='Distilled published')
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(tt, fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f'κ = {kap:g}')
            if i == 1:
                ax.set_xlabel('Round')
            panel_letter(ax, 'abcdefgh'[4 * i + j], dx=-0.2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.06), handlelength=1.4)
    fig.savefig(path, bbox_inches='tight')


def ed_optim(E, runs_es, runs_gnn, best, path):
    """Optimization diagnostics. runs_*: dict name -> log list. best: DataFrame of best-found comparison."""
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    for f in ['e3', 'welfare', 'coopfinal']:
        for nm, lg in runs_es.items():
            if re.fullmatch(f'table_{f}_k0.5_s[0-9]+', nm):
                it = [r['it'] for r in lg]; rr = [r['coop_final'] for r in lg]
                ax.plot(it, rr, color=COL[f], lw=0.6, alpha=0.8)
    ax.set_xlabel('ES iteration'); ax.set_ylabel('Training final coop.'); ax.set_title('ES, κ = 0.5')
    panel_letter(ax, 'a')
    ax = axs[1]
    for f in ['e3', 'welfare', 'coopfinal']:
        for nm, lg in runs_gnn.items():
            if re.fullmatch(f'gnn_{f}_k0.5_s[0-9]+', nm):
                ax.plot([r['it'] for r in lg], [r['coop_final'] for r in lg], color=COL[f], lw=0.6, alpha=0.8)
            if re.fullmatch(f'gnn_{f}-joint_k0.5_s[0-9]+', nm):
                ax.plot([r['it'] for r in lg], [r['coop_final'] for r in lg], color=COL[f], lw=0.6, ls='--')
    ax.set_xlabel('PPO iteration'); ax.set_ylabel('Training final coop.'); ax.set_title('Graph network, κ = 0.5')
    panel_letter(ax, 'b')
    ax = axs[2]
    for mode, ls in [('', '-'), ('-joint', '--')]:
        for nm, lg in runs_gnn.items():
            if f'{mode}_k0.5_s' in nm and ('joint' in nm) == (mode == '-joint') and 'clip_joint' in lg[-1]:
                ax.plot([r['it'] for r in lg], [r['clip_joint'] for r in lg], ls, color='#4d4d4d', lw=0.5, alpha=0.6)
                ax.plot([r['it'] for r in lg], [r['clip_comp'] for r in lg], ls, color='#d6604d', lw=0.5, alpha=0.6)
    ax.plot([], [], color='#4d4d4d', label='joint ratio outside 1 ± 0.2')
    ax.plot([], [], color='#d6604d', label='per-pair ratio outside 1 ± 0.2')
    ax.plot([], [], color='0.5', ls='--', label='joint-ratio PPO')
    ax.set_xlabel('PPO iteration'); ax.set_ylabel('Fraction of samples'); ax.set_title('Clipping diagnostics')
    ax.legend(fontsize=5.5, loc='upper right', handlelength=1.2, bbox_to_anchor=(1.05, 1.02))
    panel_letter(ax, 'c')
    ax = axs[3]
    cols = {'ES': '#252525', 'ES multi-start': '#737373', 'GNN': '#2166ac', 'GNN joint': '#67a9cf', 'hand rule': '#d6604d'}
    for j, o in enumerate(best.obj.unique()):
        sub = best[best.obj == o]
        for _, r in sub.iterrows():
            ax.plot(j + {'ES': -0.24, 'ES multi-start': -0.12, 'GNN': 0, 'GNN joint': 0.12, 'hand rule': 0.24}[r.src], r.short_pct,
                    'o', color=cols[r.src], ms=2.8)
    for k, v in cols.items():
        ax.plot([], [], 'o', color=v, ms=2.8, label=k)
    ax.set_xticks(range(best.obj.nunique())); ax.set_xticklabels([SHORT[o] for o in best.obj.unique()], rotation=45, ha='right')
    ax.set_ylabel('Shortfall from best found (%)'); ax.set_title('Best found per objective, κ = 0.5')
    ax.legend(fontsize=5.5, loc='upper left', handlelength=1)
    panel_letter(ax, 'd')
    fig.savefig(path, bbox_inches='tight')


def ed_tie(marg, coef, path):
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.55))
    pl = ['encouragement', 'es_welfare', 'es_e3', 'es_coopmean']
    pcol = {'encouragement': 'k', 'es_welfare': COL['welfare'], 'es_e3': COL['e3'], 'es_coopmean': COL['coopmean']}
    plab = {'encouragement': 'Distilled', 'es_welfare': 'Welfare', 'es_e3': 'Capital', 'es_coopmean': 'Mean coop.'}
    for ax, (m, yl, tt), L in zip(axs[:3], [('break_even', 'Break-even κ', 'Break-even depends on states'),
                                            ('ms_n_cum', 'Δ cooperator-rounds to game end', 'Multi-step cooperation'),
                                            ('ms_pay', 'Δ total payoff to game end', 'Multi-step payoff')], 'abc'):
        for p in pl:
            for k, mk in [(1.0, 'o'), (0.5, 's')]:
                sub = marg[(marg.planner == p) & (marg.kappa == k)].sort_values('t')
                lo = sub[m + '_lo'] if m + '_lo' in sub else sub[m]
                hi = sub[m + '_hi'] if m + '_hi' in sub else sub[m]
                ax.errorbar(sub.t + (0.4 if k == 0.5 else 0), sub[m], yerr=[sub[m] - lo, hi - sub[m]], fmt=mk + '-',
                            color=pcol[p], ms=2.5, lw=0.7, mfc='white' if k == 0.5 else pcol[p], capsize=0)
        ax.axhline(0 if m != 'break_even' else 1, color='0.6', lw=0.5, ls=':')
        ax.set_xlabel('Counterfactual decision round'); ax.set_ylabel(yl); ax.set_title(tt)
        ax.set_xticks([2, 7, 13])
        panel_letter(ax, L)
    for p in pl:
        axs[0].plot([], [], 'o', color=pcol[p], ms=2.5, label=plab[p])
    axs[0].plot([], [], 'o', color='0.4', ms=2.5, label='states at κ = 1'); axs[0].plot([], [], 's', color='0.4', mfc='white', ms=2.5, label='states at κ = 0.5')
    axs[0].set_ylim(0.45, 1.02)
    axs[0].legend(fontsize=5.0, loc='center', bbox_to_anchor=(0.5, 0.37), ncol=2, handlelength=1, columnspacing=0.8, frameon=False)
    ax = axs[3]
    dr = coef[coef.draw != 'point']
    for t, col in zip([2, 7, 13], ['#bdbdbd', '#737373', '#252525']):
        x = dr[dr.t == t].break_even.astype(float)
        x = x.fillna(1.6)
        ax.hist(x, bins=np.arange(0, 1.65, 0.05), histtype='step', color=col, label=f'round {t}')
    ax.axvline(1, color='0.6', lw=0.5, ls=':')
    ax.set_xlabel('Break-even κ (1.6 = none ≤ 1.5)'); ax.set_ylabel('Coefficient draws'); ax.set_title('Coefficient uncertainty')
    ax.legend(fontsize=5.5, loc='upper left', handlelength=1)
    panel_letter(ax, 'd')
    fig.savefig(path, bbox_inches='tight', bbox_extra_artists=fig.get_default_bbox_extra_artists() + [ax.xaxis.label])


def ed_calib(cal, human, path):
    apply_style()
    conds = ['static', 'random', 'clustering', 'neutral', 'encouragement', 'maxconn']
    nm = {'static': 'Static network', 'random': 'Random\nrecommendations', 'clustering': 'Cooperative\nclustering',
          'neutral': 'Neutral schedule', 'encouragement': 'Encouragement\n(distilled)', 'maxconn': 'Max. connectivity'}
    fig, axs = plt.subplots(1, 6, figsize=(7.2, 1.8), sharey=True, gridspec_kw=dict(wspace=0.15))
    base = cal[cal.cond.apply(lambda c: set(c) == {'kappa'})].copy()
    base['kappa'] = base.cond.apply(lambda c: c['kappa'])
    for ax, p, L in zip(axs, conds, 'abcdef'):
        sub = base[base.planner == p].sort_values('kappa')
        ax.plot(sub.kappa, sub.coop_final, 'o-', color='#2166ac', ms=2.5)
        ax.axhline(human[p], color='#d6604d', lw=1)
        ax.set_title(nm[p], fontsize=6)
        ax.set_xlabel('κ')
        ax.set_ylim(0, 0.9)
        panel_letter(ax, L, dx=-0.1, dy=1.10)
    axs[0].set_ylabel('Final-round cooperation')
    axs[0].plot([], [], 'o-', color='#2166ac', ms=2.5, label='simulator'); axs[0].plot([], [], color='#d6604d', label='human groups')
    axs[0].legend(fontsize=5.5, loc='lower left')
    fig.savefig(path, bbox_inches='tight')


def ed_stress(E, path):
    apply_style()
    core5 = MAIN5
    conds = [('k1', 'fitted'), ('k1_mu_th-1.304', 'μθ − 1'), ('k1_ben0.15', 'b = 0.15'), ('k0.5_ben0.15', 'b = 0.15,\nκ = 0.5'),
             ('k1_w_imit0.3_temp0.5', 'w = 0.3,\nT = 0.5'), ('k1_w_imit0.3_imitunbiased', 'w = 0.3,\nunbiased'),
             ('k0.5_w_imit0.3', 'w = 0.3,\nκ = 0.5')]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.35, width_ratios=[1.4, 1.4, 1]))
    for ax, m, yl, L in [(axs[0], 'coop_final', 'Final-round cooperation', 'a'), (axs[1], 'rawls', 'Minimum final capital', 'b')]:
        for j, (c, lab) in enumerate(conds):
            for i, f in enumerate(core5):
                sub = own(E, c, 'table', f)
                ax.plot(j + (i - 2) * 0.13, sub[m].mean(), 'o', color=COL[f], ms=2.8)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([l.replace('\n', ' ') for _, l in conds], fontsize=5.5, rotation=45, ha='right')
        ax.set_ylabel(yl)
        panel_letter(ax, L)
    axs[0].set_title('Additional behavioural and game variants')
    ax = axs[2]
    for f in core5:
        a = [own(E, c, 'table', f).coop_final.mean() for c in ['k1', 'k0.5']]
        b = [own(E, c, 'table_r', f).coop_final.mean() for c in ['k1', 'k0.5']]
        ax.plot([0, 1], a, 'o-', color=COL[f], ms=2.8)
        ax.plot([0.08, 1.08], b, 's--', color=COL[f], mfc='white', ms=2.8, lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['κ = 1', 'κ = 0.5'])
    ax.set_ylabel('Final-round cooperation'); ax.set_title('State-augmented schedules')
    ax.plot([], [], 'o-', color='0.4', ms=2.8, label='round × type × status'); ax.plot([], [], 's--', color='0.4', mfc='white', ms=2.8, label='+ defector x_r bin')
    ax.legend(fontsize=5.5, loc='lower left')
    panel_letter(ax, 'c')
    h = [Line2D([], [], color=COL[f], marker='o', ls='', ms=2.8) for f in core5]
    fig.legend(h, [SHORT[f] for f in core5], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(path, bbox_inches='tight')


def ed_esgnn(N, path):
    apply_style()
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    mk = {'k1': 'o', 'k0.75': '^', 'k0.5': 's'}
    for c, f, x, y in N['es_gnn']['pairs']:
        ax.plot(x, y, mk[c], color=COL[f], ms=3.5)
    lo, hi = 0.58, 0.84
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.6, ls=':')
    ax.set_xlabel('Schedule planner (ES), final coop.'); ax.set_ylabel('Graph network (PPO), final coop.')
    ax.set_title(f"r = {N['es_gnn']['pearson']:.2f}, MAE = {N['es_gnn']['mae']:.3f}")
    for c, lab in [('k1', 'κ = 1'), ('k0.75', 'κ = 0.75'), ('k0.5', 'κ = 0.5')]:
        ax.plot([], [], mk[c], color='0.4', ms=3.5, label=lab)
    ax.legend(fontsize=5.5, loc='upper left')
    fig.savefig(path, bbox_inches='tight')
