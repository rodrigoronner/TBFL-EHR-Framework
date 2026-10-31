"""
Gera as figuras (painéis PDF vetoriais com estilo uniforme) e o report.json com todos os
números citados no manuscrito, a partir dos CSVs de results/. Nenhum valor é sintetizado.

Cada painel é salvo como um PDF próprio com largura fixa, para ser montado no LaTeX com
subfigure/subcaption ("(a) ...", "(b) ..."). Larguras calibradas para \\textwidth = 372 pt
(template Springer Nature sn-jnl).
"""
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve

from tbfl_fl import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('TBFL_OUT', os.path.join(HERE, '..', 'results'))
CACHE = os.environ.get('TBFL_CACHE', os.path.join(HERE, '..', 'data', 'cache'))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

# ------------------------- Estilo uniforme -------------------------
TEXTWIDTH_IN = 372 / 72.27
W3 = 0.32 * TEXTWIDTH_IN          # painel em linha de três
W2 = 0.48 * TEXTWIDTH_IN          # painel em linha de dois
H = 1.45                          # altura comum dos painéis (polegadas)

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.unicode_minus': False, 'axes.formatter.use_mathtext': True,
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.2, 'lines.markersize': 3.5,
    'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Paleta categórica validada (ordem fixa; a cor segue a entidade, nunca a posição)
AGGS = ['tbfl', 'fedavg', 'median', 'trimmed', 'krum', 'foolsgold']
COLOR = {'tbfl': '#2a78d6', 'fedavg': '#eb6834', 'median': '#1baf7a',
         'trimmed': '#eda100', 'krum': '#e87ba4', 'foolsgold': '#008300'}
MARKER = {'tbfl': 'o', 'fedavg': 's', 'median': '^', 'trimmed': 'v', 'krum': 'D',
          'foolsgold': 'P'}
LABEL = {'tbfl': 'TBFL (identity gate)', 'fedavg': 'FedAvg (unprotected)',
         'median': 'Coordinate-wise median', 'trimmed': 'Trimmed mean', 'krum': 'Krum',
         'foolsgold': 'FoolsGold'}
BLUE, ORANGE, NEUTRAL = COLOR['tbfl'], COLOR['fedavg'], '#8a8984'
FLIP = '#7c4dbd'
FLIP_DEFS = ['fedavg', 'median', 'krum', 'normclip', 'foolsgold', 'tbfl']
FLIP_SHORT = {'fedavg': 'Fed-\nAvg', 'median': 'Median', 'krum': 'Krum', 'normclip': 'Norm\nclip',
              'foolsgold': 'Fools-\nGold', 'tbfl': 'TBFL'}

GAS_PRICE_GWEI = {'Ethereum L1': 20, 'Arbitrum L2': 0.1}
ETH_USD = 3000


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


def panel(w=W3, h=H):
    return plt.subplots(figsize=(w, h))


def ms(x):
    """Média ± desvio-padrão amostral, como string e como números."""
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {'mean': float(np.mean(x)), 'sd': sd, 'n': int(len(x)),
            'fmt': f'{np.mean(x):.3f} ± {sd:.3f}'}


def load(agg, S, rounds=None, div=0, suffix=''):
    tag = f'{agg}_S{S}' + (f'_div{div:g}' if div else '') + suffix
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f'{tag}_seed*.csv'))):
        if f.endswith(('_chain.csv', '_clients.csv')):
            continue
        d = pd.read_csv(f)
        if rounds is None or len(d) == rounds:
            d['seed'] = int(f.rsplit('seed', 1)[1].split('.')[0])
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_baseline(mode):
    fs = sorted(glob.glob(os.path.join(RES, 'baselines', f'{mode}_seed*.csv')))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).assign(seed=int(f.rsplit('seed', 1)[1].split('.')[0]))
                      for f in fs], ignore_index=True)


def band(ax, df, col, color, label=None, marker=None, every=10):
    g = df.groupby('round')[col]
    m, s = g.mean(), g.std().fillna(0)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.18, lw=0)
    ax.plot(m.index, m.values, color=color, label=label, marker=marker,
            markevery=every, markeredgewidth=0)


def legend_strip(handles, name, ncol):
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.28))
    fig.legend(handles=handles, loc='center', ncol=ncol, handlelength=2.2, columnspacing=1.4)
    save(fig, name)


def agg_handles(aggs):
    return [Line2D([], [], color=COLOR[a], marker=MARKER[a], markeredgewidth=0,
                   label=LABEL[a]) for a in aggs]


def main():
    rep = {}

    # ===================== Dados e federação (seed 0) =====================
    meta = json.load(open(os.path.join(RES, 'tbfl_S0_seed0_meta.json')))
    raw = np.array(meta['client_raw']) / 0.8          # treino = 80% de cada cliente
    pos = np.array(meta['client_pos_rate_raw']) * 100
    prev = meta['test_pos'] / meta['test_n']
    k = np.arange(1, len(raw) + 1)
    rep['dataset'] = {'features': meta['features'], 'n_features': len(meta['features']),
                      'client_admissions_approx': raw.round().astype(int).tolist(),
                      'client_mortality_pct': pos.round(2).tolist(),
                      'client_train_balanced': meta['client_sizes_balanced'],
                      'test_n': meta['test_n'], 'test_pos': meta['test_pos'],
                      'deploy_gas': meta['deploy_gas']}
    fig, ax = panel(W2)
    ax.bar(k, raw / 1e3, color=BLUE, width=0.7)
    ax.set_xlabel('Client'); ax.set_ylabel(r'Admissions ($\times 10^3$)'); ax.set_xticks(k)
    ax.grid(axis='x', visible=False)
    save(fig, 'data_a_sizes')
    fig, ax = panel(W2)
    ax.bar(k, pos, color=BLUE, width=0.7)
    ax.axhline(prev * 100, color=INK2, ls='--', lw=0.8)
    ax.text(10.45, prev * 100 + 0.3, 'pooled', ha='right', color=INK2, fontsize=7)
    ax.set_xlabel('Client'); ax.set_ylabel('In-hospital mortality (%)'); ax.set_xticks(k)
    ax.grid(axis='x', visible=False)
    save(fig, 'data_b_mortality')

    # ===================== Exp. 1: desempenho sem ataque =====================
    main_ = load('tbfl', 0, rounds=100)
    final = main_[main_['round'] == 100]
    cols = ['auc', 'auprc', 'brier', 'accuracy', 'precision', 'recall', 'f1', 'fnr']
    rep['main_final'] = {c: ms(final[c]) for c in cols}
    best = main_.loc[main_.groupby('seed')['auc'].idxmax()]
    rep['main_best_round'] = {'auc': ms(best['auc']), 'rounds': best['round'].tolist()}
    rep['main_per_seed_final'] = final.set_index('seed')[['auc', 'auprc', 'recall']] \
        .round(4).to_dict('index')

    cen = load_baseline('central_fed')
    for col, name, ylab in [('auc', 'conv_a_auc', 'AUC-ROC'), ('auprc', 'conv_b_auprc', 'AUPRC'),
                            ('brier', 'conv_c_brier', 'Brier score')]:
        fig, ax = panel()
        band(ax, main_, col, BLUE)
        if col == 'auprc':
            ax.axhline(prev, color=NEUTRAL, ls=':', lw=0.9)
        if cen is not None:
            ax.axhline(cen[cen['epoch'] == cen['epoch'].max()][col].mean(), color=INK2,
                       ls='--', lw=0.9)
        ax.set_xlabel('Communication round'); ax.set_ylabel(ylab); ax.set_xlim(0, 100)
        save(fig, name)
    legend_strip([Line2D([], [], color=BLUE, label='TBFL, no attack (mean ± SD, 5 seeds)'),
                  Line2D([], [], color=INK2, ls='--', label='Centralized training'),
                  Line2D([], [], color=NEUTRAL, ls=':', label='Random classifier (AUPRC)')],
                 'conv_legend', ncol=3)

    # Matriz de confusão, ROC e PR do modelo final
    r0 = final[final['seed'] == 0].iloc[0]
    cm = np.array([[r0.tn, r0.fp], [r0.fn, r0.tp]], dtype=int)
    rep['confusion_seed0'] = {'tn': int(r0.tn), 'fp': int(r0.fp), 'fn': int(r0.fn),
                              'tp': int(r0.tp)}
    fig, ax = panel()
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f'{cm[i, j]:,}\n({frac * 100:.1f}%)', ha='center',
                    va='center', fontsize=7, color='white' if frac > 0.5 else INK)
    ax.set_xticks([0, 1], ['Survived', 'Died']); ax.set_yticks([0, 1], ['Survived', 'Died'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'clin_a_confusion')

    curves = {'roc': [], 'pr': []}
    for s in sorted(final['seed'].unique()):
        mp = os.path.join(RES, f'tbfl_S0_seed{s}_model.pt')
        cp = os.path.join(CACHE, f'fed_k10_a0.5_s{s}.pkl')
        if not (os.path.exists(mp) and os.path.exists(cp)):
            continue
        _, X_test, y_test, names = pickle.load(open(cp, 'rb'))
        model = MLP(len(names)); model.load_state_dict(torch.load(mp)); model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.as_tensor(X_test)).squeeze(1)).numpy()
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        curves['roc'].append((fpr, tpr)); curves['pr'].append((rec, prec))
    if curves['roc']:
        fig, ax = panel()
        for fpr, tpr in curves['roc']:
            ax.plot(fpr, tpr, color=BLUE, lw=0.9, alpha=0.7)
        ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=':', lw=0.9)
        ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        save(fig, 'clin_b_roc')
        fig, ax = panel()
        for rec, prec in curves['pr']:
            ax.plot(rec, prec, color=BLUE, lw=0.9, alpha=0.7)
        ax.axhline(prev, color=NEUTRAL, ls=':', lw=0.9)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        save(fig, 'clin_c_pr')
        rep['roc_pr_seeds'] = len(curves['roc'])

    # ===================== Baselines =====================
    rep['baselines'] = {}
    for mode in ['central_fed', 'central_adm', 'central_pat']:
        d = load_baseline(mode)
        if d is None:
            continue
        f = d[d['epoch'] == d['epoch'].max()]
        rep['baselines'][mode] = {c: ms(f[c]) for c in ['auc', 'auprc', 'brier', 'precision',
                                                        'recall', 'f1']}
    loc = load_baseline('local')
    cl_files = sorted(glob.glob(os.path.join(RES, 'tbfl_S0_seed*_clients.csv')))
    if loc is not None and cl_files:
        rep['baselines']['local_on_own_test'] = {
            'auc': ms(loc.groupby('seed')['local_auc'].mean()),
            'auprc': ms(loc.groupby('seed')['local_auprc'].mean())}
        rep['baselines']['local_on_global_test'] = {
            'auc': ms(loc.groupby('seed')['global_auc'].mean()),
            'auprc': ms(loc.groupby('seed')['global_auprc'].mean())}
        fl = pd.concat([pd.read_csv(f).assign(seed=int(f.split('seed')[1].split('_')[0]))
                        for f in cl_files])
        fl = fl[fl['round'] == fl['round'].max()]
        rep['baselines']['fl_on_client_test'] = {'auc': ms(fl.groupby('seed')['auc'].mean()),
                                                 'auprc': ms(fl.groupby('seed')['auprc'].mean())}
        l0 = loc[loc['seed'] == 0].set_index('client')
        f0 = fl[fl['seed'] == 0].set_index('client')
        x = np.arange(1, len(l0) + 1); w = 0.38
        for col, name, ylab in [('auc', 'clients_a_auc', 'AUC-ROC'),
                                ('auprc', 'clients_b_auprc', 'AUPRC')]:
            fig, ax = panel(W2)
            ax.bar(x - w / 2, f0[col], w, color=BLUE)
            ax.bar(x + w / 2, l0['local_' + col], w, color=ORANGE)
            ax.set_xticks(x); ax.set_xlabel('Client (local test set)'); ax.set_ylabel(ylab)
            if col == 'auc':
                ax.set_ylim(0.5, 1)
            ax.grid(axis='x', visible=False)
            save(fig, name)
        legend_strip([Patch(color=BLUE, label='Federated model (TBFL)'),
                      Patch(color=ORANGE, label='Model trained on the client alone')],
                     'clients_legend', ncol=2)
        rep['clients_seed0'] = pd.DataFrame({
            'mortality_pct': (f0['pos_rate'] * 100).round(2), 'fl_auc': f0['auc'].round(4),
            'local_auc': l0['local_auc'].round(4), 'fl_auprc': f0['auprc'].round(4),
            'local_auprc': l0['local_auprc'].round(4)}).reset_index().to_dict('records')

    # ===================== Exp. 2/3: Sybil =====================
    rows, curves_s = [], {}
    for div in [0, 1]:
        for S in [2, 5, 10, 15]:
            for agg in AGGS + ['normclip', 'tbfl-median']:
                d = load(agg, S, rounds=50, div=div)
                if d.empty:
                    continue
                f = d[d['round'] == 50]
                att = d[d['round'] >= 10]
                share = (float(att['sybil_weight_share'].mean())
                         if 'sybil_weight_share' in att else None)
                rows.append({'div': div, 'sybils': S, 'share': S / (S + 10), 'agg': agg,
                             'n': len(f), 'auc': ms(f['auc']), 'auprc': ms(f['auprc']),
                             'recall': ms(f['recall']), 'fnr': ms(f['fnr']),
                             'rejected_per_round': float(f['n_rejected'].mean()),
                             'sybil_weight_share': share})
                if agg in AGGS:
                    curves_s[(agg, S, div)] = d
    rep['sybil'] = rows

    # Passo 2: filtro + mediana, corte por norma, variação de sigma e rótulos invertidos
    def final_auc(agg, S, div=0, suffix=''):
        d = load(agg, S, rounds=50, div=div, suffix=suffix)
        return None if d.empty else {c: ms(d[d['round'] == 50][c]) for c in ['auc', 'auprc']}
    rep['step2'] = {
        'gate_median': {S: final_auc('tbfl-median', S) for S in [5, 10, 15]},
        'normclip': {S: final_auc('normclip', S) for S in [0, 2, 5, 10, 15]},
        'sigma': {f'{a}_s{dv:g}': final_auc(a, 10, div=dv)
                  for a in ['median', 'krum', 'foolsgold'] for dv in [0, 0.1, 0.3, 1]},
        'labelflip': {f'{a}_S{S}': final_auc(a, S, suffix='_flip')
                      for a in ['fedavg', 'median', 'krum', 'foolsgold', 'normclip', 'tbfl']
                      for S in [5, 10]},
        'clean_r50': {a: final_auc(a, 0) for a in ['fedavg', 'median', 'trimmed', 'krum',
                                                   'foolsgold', 'normclip']}}

    # Parada antecipada: rodada escolhida pela AUC de validação versus a rodada final
    es = load('fedavg', 0, rounds=100, suffix='_val0.1')
    if not es.empty:
        sel = es.loc[es.groupby('seed')['val_auc'].idxmax()]
        fin = es[es['round'] == 100]
        rep['early_stop'] = {
            'selected_rounds': sel['round'].astype(int).tolist(),
            'selected': {c: ms(sel[c]) for c in ['auc', 'auprc', 'brier', 'recall']},
            'final': {c: ms(fin[c]) for c in ['auc', 'auprc', 'brier', 'recall']}}

    # Tentativas de Sybils e falhas de clientes honestos no ledger durante o treino
    chain = pd.concat([pd.read_csv(f).assign(run=os.path.basename(f))
                       for f in glob.glob(os.path.join(RES, 'tbfl*_chain.csv'))])
    att = chain[chain['kind'].str.startswith('sybil')]
    hon = chain[chain['kind'] == 'submit']
    rep['ledger_attempts'] = {
        'runs_under_attack': int(att['run'].nunique()),
        'sybil': att.groupby('kind')['accepted'].agg(['sum', 'size']).astype(int)
        .to_dict('index'),
        'honest_submit': {'failed': int((~hon['accepted']).sum()), 'total': int(len(hon))}}
    d = load('fedavg', 0, rounds=50)
    if not d.empty:
        f = d[d['round'] == 50]
        rep['fedavg_clean_r50'] = {c: ms(f[c]) for c in ['auc', 'auprc', 'recall', 'fnr']}
    f = main_[main_['round'] == 50]
    rep['tbfl_clean_r50'] = {c: ms(f[c]) for c in ['auc', 'auprc', 'recall', 'fnr']}

    present = [a for a in AGGS if any(k_[0] == a and k_[2] == 0 for k_ in curves_s)]
    for S, lab in zip([2, 5, 10, 15], 'abcd'):
        fig, ax = panel(W2, 1.55)
        for agg in present:
            if (agg, S, 0) in curves_s:
                band(ax, curves_s[(agg, S, 0)], 'auc', COLOR[agg], marker=MARKER[agg], every=8)
        ax.axvline(10, color=INK2, ls=':', lw=0.8)
        ax.set_xlabel('Communication round'); ax.set_ylabel('AUC-ROC')
        ax.set_ylim(0, 1); ax.set_xlim(1, 50)
        save(fig, f'sybil_{lab}_S{S}')
    legend_strip(agg_handles(present), 'sybil_legend', ncol=3)

    syb = pd.DataFrame([{'div': r['div'], 'sybils': r['sybils'], 'share': r['share'],
                         'agg': r['agg'], 'auc': r['auc']['mean'], 'sd': r['auc']['sd']}
                        for r in rows])
    if not syb.empty:
        fig, ax = panel(W2, 1.75)
        for agg in present:
            s = syb[(syb['agg'] == agg) & (syb['div'] == 0)].sort_values('share')
            ax.errorbar(s['share'] * 100, s['auc'], yerr=s['sd'], color=COLOR[agg],
                        marker=MARKER[agg], capsize=2, elinewidth=0.7, markeredgewidth=0)
        ax.axvline(50, color=INK2, ls=':', lw=0.8)
        ax.set_xlabel('Sybil identities (% of participants)')
        ax.set_ylabel('AUC-ROC (round 50)'); ax.set_ylim(0, 1)
        save(fig, 'frac_a_identical')
        adv = syb[syb['div'] == 1]
        if not adv.empty:
            fig, ax = panel(W2, 1.75)
            defs = [a for a in ['foolsgold', 'median', 'trimmed', 'krum'] if a in set(adv['agg'])]
            x = np.arange(len(defs)); w = 0.2
            for i, (S, dv) in enumerate([(5, 0), (5, 1), (10, 0), (10, 1)]):
                v = [syb[(syb['agg'] == a) & (syb['sybils'] == S) & (syb['div'] == dv)]['auc']
                     .mean() for a in defs]
                ax.bar(x + (i - 1.5) * w, v, w, color=[BLUE, ORANGE][dv],
                       hatch=None if S == 5 else '////', edgecolor='white', lw=0)
            short = {'foolsgold': 'FoolsGold', 'median': 'Median', 'trimmed': 'Trimmed', 'krum': 'Krum'}
            ax.set_xticks(x, [short[a] for a in defs])
            ax.set_ylabel('AUC-ROC (round 50)'); ax.set_ylim(0, 1)
            ax.grid(axis='x', visible=False)
            save(fig, 'frac_b_adaptive')
            legend_strip([Patch(color=BLUE, label='Identical Sybils'),
                          Patch(color=ORANGE, label='Adaptive Sybils'),
                          Patch(color=FLIP, label='Label-flipping Sybils'),
                          Patch(facecolor='white', edgecolor=INK2, label='33% Sybils (solid)'),
                          Patch(facecolor='white', edgecolor=INK2, hatch='////',
                                label='50% Sybils (hatched)')], 'frac_legend', ncol=5)
        flip = {(a, S): load(a, S, rounds=50, suffix='_flip') for a in FLIP_DEFS for S in [5, 10]}
        if all(not d.empty for d in flip.values()):
            fig, ax = panel(W2, 1.75)
            x = np.arange(len(FLIP_DEFS)); w = 0.36
            for i, S in enumerate([5, 10]):
                v = [flip[(a, S)][flip[(a, S)]['round'] == 50]['auc'].mean() for a in FLIP_DEFS]
                ax.bar(x + (i - 0.5) * w, v, w, color=FLIP, hatch=None if S == 5 else '////',
                       edgecolor='white', lw=0)
            ax.set_xticks(x, [FLIP_SHORT[a] for a in FLIP_DEFS], fontsize=6.5)
            ax.set_ylabel('AUC-ROC (round 50)'); ax.set_ylim(0, 1)
            ax.grid(axis='x', visible=False)
            save(fig, 'frac_d_labelflip')
        if ('foolsgold', 5, 0) in curves_s and ('foolsgold', 5, 1) in curves_s:
            fig, ax = panel(W2, 1.75)
            for dv, c, lab in [(0, BLUE, 'Identical Sybils'), (1, ORANGE, 'Adaptive Sybils')]:
                d = curves_s[('foolsgold', 5, dv)]
                band(ax, d[d['round'] >= 10], 'sybil_weight_share', c, label=lab, every=8)
            ax.set_xlabel('Communication round'); ax.set_ylabel('Weight given to Sybils')
            ax.set_ylim(-0.02, 1.02); ax.set_xlim(10, 50); ax.legend(loc='center right')
            save(fig, 'frac_c_foolsgold_weight')

    # ===================== Custo e sobrecarga =====================
    ch = pd.read_csv(os.path.join(RES, 'tbfl_S0_seed0_chain.csv'))
    acc = ch[ch['accepted']]
    rep['gas_by_kind'] = acc.groupby('kind')['gas'].agg(['mean', 'min', 'max', 'count']) \
        .round(1).to_dict('index')
    cum = acc.groupby('round')['gas'].sum().cumsum()
    per_round = float(acc[acc['kind'] == 'submit'].groupby('round')['gas'].sum().mean())
    total = int(cum.iloc[-1])
    rep['gas_per_round_submissions'] = per_round
    rep['gas_total_100_rounds'] = total
    rep['gas_linear_r2'] = round(stats.linregress(cum.index, cum.values).rvalue ** 2, 6)
    rep['cost_table'] = {net: {'usd_per_round': round(per_round * g * 1e-9 * ETH_USD, 4),
                               'usd_100_rounds': round(total * g * 1e-9 * ETH_USD, 2)}
                         for net, g in GAS_PRICE_GWEI.items()}
    tt = main_.groupby('seed')['avg_train_time'].mean()
    ct = main_.groupby('seed')['avg_chain_time'].mean()
    rep['train_time_s'] = ms(tt); rep['chain_time_ms'] = ms(ct * 1e3)
    rep['overhead_pct'] = ms(ct / tt * 100)

    bench_p = os.path.join(RES, 'bench_registry.csv')
    if os.path.exists(bench_p):
        b = pd.read_csv(bench_p)
        rep['bench'] = b.groupby(['contract', 'op']).agg(
            n=('gas', 'size'), accepted=('accepted', 'sum'), gas_mean=('gas', 'mean'),
            gas_median=('gas', 'median'), gas_max=('gas', 'max'),
            latency_ms=('latency_s', lambda s: 1e3 * s.mean())
        ).round(1).reset_index().to_dict('records')
        ok = b[b['accepted']]
        variants = ['FLRegistry', 'FLRegistryEvents', 'FLRegistryZK']
        vlabel = {'FLRegistry': 'Allowlist', 'FLRegistryEvents': 'Allowlist with events',
                  'FLRegistryZK': 'Signed credential (EIP-712)'}
        vcol = [BLUE, COLOR['median'], ORANGE]
        fig, ax = panel()
        x = np.arange(2); w = 0.26
        for i, v in enumerate(variants):
            vals = [ok[(ok['contract'] == v) & (ok['op'] == o)]['gas'].median() / 1e3
                    for o in ['authorize', 'submit']]
            ax.bar(x + (i - 1) * w, vals, w, color=vcol[i])
        ax.set_xticks(x, ['Authorise', 'Submit']); ax.set_ylabel(r'Gas ($\times 10^3$)')
        ax.grid(axis='x', visible=False)
        save(fig, 'cost_a_gas_ops')
        legend_strip([Patch(color=c, label=vlabel[v]) for v, c in zip(variants, vcol)],
                     'cost_legend', ncol=3)
    fig, ax = panel()
    for net, c in zip(GAS_PRICE_GWEI, [ORANGE, BLUE]):
        ax.plot(cum.index, cum.values * GAS_PRICE_GWEI[net] * 1e-9 * ETH_USD, color=c, label=net)
    ax.set_yscale('log'); ax.set_xlabel('Communication round')
    ax.set_ylabel(r'Cumulative cost (US\$)'); ax.legend(loc='lower right'); ax.set_xlim(0, 100)
    save(fig, 'cost_b_cumulative')
    fig, ax = panel()
    vals = [tt.mean(), ct.mean()]
    ax.bar([0, 1], vals, color=[BLUE, ORANGE], width=0.6)
    ax.set_yscale('log'); ax.set_xticks([0, 1], ['Local\ntraining', 'On-chain\ncommitment'])
    ax.set_ylabel('Time per client and round (s)'); ax.grid(axis='x', visible=False)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.3, f'{v:.2f} s' if v >= 1 else f'{v * 1e3:.0f} ms', ha='center',
                fontsize=7, color=INK)
    ax.set_ylim(0.01, 30)
    save(fig, 'cost_c_time')

    json.dump(rep, open(os.path.join(FIG, 'report.json'), 'w'), indent=1, default=str)
    print(json.dumps({k_: v for k_, v in rep.items() if k_ not in ('dataset', 'bench', 'sybil')},
                     indent=1, default=str)[:6000])
    print(pd.DataFrame([{'div': r['div'], 'S': r['sybils'], 'agg': r['agg'], 'n': r['n'],
                         'auc': r['auc']['fmt'], 'w_syb': r['sybil_weight_share']}
                        for r in rows]).to_string())


if __name__ == '__main__':
    main()
