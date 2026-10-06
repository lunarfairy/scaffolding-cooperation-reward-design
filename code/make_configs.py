"""Generate the configuration lists of the revision experiments (configs_es2.json, configs_gnn2.json)."""
import json

ES = dict(iters=400, npair=256, games=32, sigma=0.5, lr=0.1)
GNN = dict(iters=600, batch=8192, lr=1e-3, ent=1e-3, epochs=2, minibatches=16)
OBJ = {'e3': dict(obj='e3', P=1.0, gamma=0.99), 'welfare': dict(obj='welfare'), 'coopmean': dict(obj='coop_mean'),
       'coopfinal': dict(obj='coop_final'), 'rawls': dict(obj='rawls'), 'sustain': dict(obj='sustain'),
       'desert': dict(obj='omega', omega=0.0), 'omega0.25': dict(obj='omega', omega=0.25),
       'omega0.5': dict(obj='omega', omega=0.5), 'omega0.75': dict(obj='omega', omega=0.75),
       'e3P0': dict(obj='e3', P=0.0, gamma=0.99), 'welfarePm': dict(obj='welfare', P='match_inc'),
       'coopmeanPm': dict(obj='coop_mean', P='match_coop')}
CORE7 = ['e3', 'welfare', 'coopmean', 'coopfinal', 'rawls', 'sustain', 'desert']
CORE5 = ['e3', 'welfare', 'coopmean', 'coopfinal', 'desert']


def cond(kappa, **kw):
    c = dict(kappa=kappa)
    c.update(kw)
    tag = f'k{kappa}'
    if 'w_imit' in kw:
        tag += f"_w{kw['w_imit']}"
    if 'temp' in kw:
        tag += f"_t{kw['temp']}"
    if kw.get('imit') == 'unbiased':
        tag += '_unb'
    if isinstance(kw.get('delta'), str):
        tag += '_' + kw['delta'].split('_')[0]
    if 'ben' in kw:
        tag += f"_b{kw['ben']}"
    if 'mu_th' in kw:
        tag += f"_mu{kw['mu_th']}"
    c['cond'] = tag
    return c


def make(policy, tag, c, seed, extra=None, suffix=''):
    d = dict(policy=policy, seed=seed, tag=tag, gamma=1.0, P=0.0)
    d.update(ES if policy in ('table', 'table_r') else GNN)
    d.update(OBJ[tag])
    d.update(c)
    if extra:
        d.update(extra)
    pre = {'table': 'table', 'table_r': 'tabler', 'gnn': 'gnn'}[policy]
    d['name'] = f"{pre}_{tag}{suffix}_{c['cond']}_s{seed}"
    return d


es, gnn = [], []
# N1 finer kappa grid
for k in [0.625, 0.875]:
    for o in CORE7:
        es += [make('table', o, cond(k), s) for s in range(5)]
    for o in ['omega0.25', 'omega0.5', 'omega0.75']:
        es += [make('table', o, cond(k), s) for s in range(3)]
# N2 propensity-matched variants
for c in [cond(0.5, delta='slopeM_0.5'), cond(1.0, delta='levelM_0.5')]:
    for o in CORE5:
        es += [make('table', o, c, s) for s in range(5)]
# N3 benefit-cost ratio b = 0.15 (c/b = 1/3)
for k in [0.5, 1.0]:
    c = cond(k, ben=0.15)
    for o in ['desert', 'omega0.25', 'omega0.5', 'omega0.75', 'welfare', 'e3', 'coopmean', 'coopfinal']:
        es += [make('table', o, c, s) for s in range(3)]
# N3b less cooperative population (mean disposition shifted by -1)
for o in CORE5:
    es += [make('table', o, cond(1.0, mu_th=-1.304), s) for s in range(3)]
# N4 imitation: strength, temperature, payoff-unbiased control, kappa x imitation
for c in [cond(1.0, w_imit=0.1), cond(1.0, w_imit=0.5), cond(1.0, w_imit=0.3, temp=0.5),
          cond(1.0, w_imit=0.3, imit='unbiased'), cond(0.5, w_imit=0.3)]:
    for o in CORE5:
        es += [make('table', o, c, s) for s in range(3)]
# N5 reward-accounting factorial (level/increment x penalty) at kappa = 0.5 and under imitation
for c in [cond(0.5), cond(1.0, w_imit=0.3)]:
    for o in ['e3P0', 'welfarePm', 'coopmeanPm']:
        es += [make('table', o, c, s) for s in range(5)]
for o in ['e3', 'welfare']:
    es += [make('table', o, cond(1.0, w_imit=0.3), s) for s in [3, 4]]
# N6 multi-start ES from hand-specified schedules
for k in [0.5, 1.0]:
    for o in CORE7:
        for init in ['conciliation', 'exclusion', 'encouragement']:
            es.append(make('table', o, cond(k), 0, dict(init=init), suffix=f'-init{init[:4]}'))
# N7 late exclusion imposed (distilled C-D rows for rounds 8-14), remainder re-optimised
for k in [0.5, 0.75, 1.0]:
    for o in ['e3', 'welfare']:
        es += [make('table', o, cond(k), s, dict(freeze='late_cd'), suffix='-latecd') for s in range(3)]
# N8 state-augmented schedule class
for k in [0.5, 1.0]:
    for o in CORE5:
        es += [make('table_r', o, cond(k), s) for s in range(3)]

# GNN: seeds 2-4 for the seven core objectives at kappa in {0.5, 1}; imitation arm; joint-ratio PPO control
for k in [0.5, 1.0]:
    for o in CORE7:
        gnn += [make('gnn', o, cond(k), s) for s in [2, 3, 4]]
for o in ['e3', 'welfare', 'coopmean', 'coopfinal']:
    gnn += [make('gnn', o, cond(1.0, w_imit=0.3), s) for s in [0, 1, 2]]
for o in ['e3', 'welfare', 'coopmean', 'coopfinal']:
    gnn += [make('gnn', o, cond(0.5), s, dict(ppo_mode='joint'), suffix='-joint') for s in [0, 1]]
assert len({c['name'] for c in es + gnn}) == len(es + gnn)
json.dump(es, open('configs_es2.json', 'w'), indent=0)
json.dump(gnn, open('configs_gnn2.json', 'w'), indent=0)
print(len(es), len(gnn))
