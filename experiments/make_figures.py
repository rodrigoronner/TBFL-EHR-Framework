"""
Gera figuras, tabelas e estatísticas a partir dos CSVs de results/ (v2).
Nenhum valor é sintetizado: tudo vem das execuções registradas.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('TBFL_OUT', os.path.join(HERE, '..', 'results'))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300,
                     'savefig.bbox': 'tight'})

GAS_PRICE_GWEI = {'Ethereum L1': 20, 'Arbitrum L2': 0.1}
ETH_USD = 3000
AGG_LABEL = {'tbfl': 'TBFL (identity-gated FedAvg)', 'fedavg': 'FedAvg (no verification)',
             'median': 'Coordinate-wise Median', 'trimmed': 'Trimmed Mean', 'krum': 'Krum'}
AGG_COLOR = {'tbfl': '#1f4e9c', 'fedavg': '#c0392b', 'median': '#8e44ad',
             'trimmed': '#d68910', 'krum': '#239b56'}


def load(agg, S, rounds=None):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f'{agg}_S{S}_seed*.csv'))):
        if f.endswith(('_chain.csv', '_clients.csv')):
            continue
        d = pd.read_csv(f)
        if rounds is None or len(d) == rounds:
            d['seed'] = int(f.split('seed')[1].split('.')[0])
            out.append(d)
    return pd.concat(out) if out else pd.DataFrame()


def band(ax, df, col, label, color, ls='-'):
    g = df.groupby('round')[col]
    m, s = g.mean(), g.std().fillna(0)
    ax.plot(m.index, m.values, ls, color=color, label=label, lw=1.8)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.2)


def fmt(x):
    return f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}' if len(x) > 1 else f'{x[0]:.3f}'


def main():
    report = {}
    main = load('tbfl', 0, rounds=100)
    final = main[main['round'] == 100]

    # ---- Tabela de desempenho clínico (Exp. 1) ----
    cols = ['auc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'fnr', 'brier']
    report['main_final'] = {c: fmt(final[c].values) for c in cols}
    report['main_seeds'] = int(final['seed'].nunique())

    # Fig. 4 — convergência (AUC-ROC e AUPRC)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    band(ax[0], main, 'auc', 'AUC-ROC', '#1f4e9c')
    band(ax[1], main, 'auprc', 'AUPRC', '#1f4e9c')
    prev = main['tp'].iloc[0] + main['fn'].iloc[0]
    prev /= (main[['tp', 'fn', 'tn', 'fp']].iloc[0].sum())
    ax[1].axhline(prev, color='gray', ls=':', label=f'Prevalence ({prev:.3f})')
    for a, t in zip(ax, ['AUC-ROC', 'AUPRC']):
        a.set_xlabel('Communication round'); a.set_ylabel(t); a.grid(alpha=0.3)
    ax[1].legend()
    fig.savefig(os.path.join(FIG, 'fig4_convergence.png')); plt.close(fig)

    # Fig. 6 — métricas no limiar 0,5
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in zip(['precision', 'recall', 'f1', 'auc'],
                      ['#d68910', '#c0392b', '#239b56', '#1f4e9c']):
        band(ax, main, c, c.upper() if c == 'auc' else c.capitalize(), col)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Score'); ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(os.path.join(FIG, 'fig6_metrics.png')); plt.close(fig)

    # Fig. 7 — matriz de confusão real (seed 0, rodada 100)
    r0 = final[final['seed'] == 0].iloc[0]
    cm = np.array([[r0.tn, r0.fp], [r0.fn, r0.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax.set_xticks([0, 1], ['Pred. survival', 'Pred. death'])
    ax.set_yticks([0, 1], ['Survival', 'Death'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    fig.savefig(os.path.join(FIG, 'fig7_confusion_matrix.png')); plt.close(fig)
    report['confusion_seed0'] = {'tn': int(r0.tn), 'fp': int(r0.fp),
                                 'fn': int(r0.fn), 'tp': int(r0.tp)}

    # Fig. 5 — heterogeneidade: AUC do modelo global no teste local de cada cliente
    cl = pd.read_csv(os.path.join(RES, 'tbfl_S0_seed0_clients.csv'))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = plt.get_cmap('tab10')
    for k, g in cl.groupby('client'):
        pr = g['pos_rate'].iloc[0]
        ax.plot(g['round'], g['auc'], '--', color=cmap(k), alpha=0.8, lw=1.2,
                label=f'Client {k + 1} (mortality {pr * 100:.1f}%)')
    gm = main[main['seed'] == 0]
    ax.plot(gm['round'], gm['auc'], color='black', lw=2.5, label='Global test set')
    ax.set_xlabel('Communication round'); ax.set_ylabel('AUC-ROC'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.savefig(os.path.join(FIG, 'fig5_heterogeneity.png')); plt.close(fig)
    last = cl[cl['round'] == cl['round'].max()]
    report['client_auc_final_seed0'] = last[['client', 'auc', 'auprc', 'pos_rate',
                                             'n_test']].round(4).to_dict('records')

    # ---- Overhead e custo (logs on-chain reais) ----
    chains = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'tbfl_S0_seed*_chain.csv'))]
    ch = pd.concat(chains)
    acc = ch[ch['accepted']]
    gas_by_kind = acc.groupby('kind')['gas'].agg(['mean', 'std', 'min', 'max']).round(1)
    report['gas_by_kind'] = gas_by_kind.to_dict('index')
    report['latency_ms_by_kind'] = (ch.groupby('kind')['latency_s'].mean() * 1000).round(2).to_dict()
    tr = main.groupby('seed')[['avg_train_time', 'avg_chain_time']].mean()
    overhead = (tr['avg_chain_time'] / tr['avg_train_time'] * 100).values
    report['train_time_s'] = fmt(tr['avg_train_time'].values)
    report['chain_time_ms'] = fmt(tr['avg_chain_time'].values * 1000)
    report['overhead_pct'] = fmt(overhead)

    fig, ax = plt.subplots(figsize=(7, 4))
    g = main.groupby('round')[['avg_train_time', 'avg_chain_time']].mean()
    ax.bar(g.index - 0.2, g['avg_train_time'], 0.4, label='Local training (off-chain)',
           color='#1f4e9c')
    ax.bar(g.index + 0.2, g['avg_chain_time'], 0.4, label='On-chain submission',
           color='#d68910')
    ax.set_yscale('log'); ax.set_xlabel('Communication round')
    ax.set_ylabel('Time per client per round (s, log scale)'); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, 'fig8_overhead.png')); plt.close(fig)

    # Custo cumulativo medido (seed 0): credenciamento + submissões
    c0 = pd.read_csv(os.path.join(RES, 'tbfl_S0_seed0_chain.csv'))
    per_round = c0[c0['accepted']].groupby('round')['gas'].sum().sort_index()
    cum = per_round.cumsum()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(cum.index, cum / 1e6, color='#1f4e9c', lw=2)
    ax1.set_xlabel('Communication round'); ax1.set_ylabel('Cumulative gas (millions)')
    ax2 = ax1.twinx()
    usd = cum * GAS_PRICE_GWEI['Ethereum L1'] * 1e-9 * ETH_USD
    ax2.plot(cum.index, usd, '--', color='#239b56', lw=2)
    ax2.set_ylabel('Equivalent cost on Ethereum L1 (USD)')
    ax1.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, 'fig9_gas_cost.png')); plt.close(fig)
    slope, intercept, rval, _, _ = stats.linregress(cum.index, cum.values)
    total_gas = int(cum.iloc[-1])
    report['gas_total_100_rounds'] = total_gas
    report['gas_linear_r2'] = round(rval ** 2, 6)
    report['gas_per_round_submissions'] = float(c0[(c0['kind'] == 'submit')]
                                                .groupby('round')['gas'].sum().mean())
    report['cost_table'] = {}
    for net, gwei in GAS_PRICE_GWEI.items():
        per_r = report['gas_per_round_submissions'] * gwei * 1e-9 * ETH_USD
        report['cost_table'][net] = {'usd_per_round': round(per_r, 4),
                                     'usd_100_rounds_total': round(
                                         total_gas * gwei * 1e-9 * ETH_USD, 2)}

    # ---- Experimento 2: Sybil ----
    rows, curves = [], {}
    for S in [2, 5, 10, 15]:
        for agg in ['fedavg', 'median', 'trimmed', 'krum', 'tbfl']:
            d = load(agg, S, rounds=50)
            if d.empty:
                continue
            f = d[d['round'] == 50]
            rows.append({'sybils': S, 'fraction': S / (S + 10), 'agg': agg,
                         'auc_mean': f['auc'].mean(), 'auc_sd': f['auc'].std(),
                         'auprc_mean': f['auprc'].mean(), 'recall_mean': f['recall'].mean(),
                         'fnr_mean': f['fnr'].mean(), 'n': len(f),
                         'rejected_per_round': f['n_rejected'].mean()})
            curves[(agg, S)] = d
    clean = load('fedavg', 0, rounds=50)
    syb = pd.DataFrame(rows)
    syb.round(4).to_csv(os.path.join(FIG, 'table_sybil.csv'), index=False)
    if not clean.empty:
        report['clean_fedavg_r50'] = {c: fmt(clean[clean['round'] == 50][c].values)
                                      for c in ['auc', 'auprc', 'recall', 'fnr']}

    # Fig. 10 — curvas sob ataque (S = 5, 33% das identidades)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for agg in ['tbfl', 'fedavg', 'median', 'trimmed', 'krum']:
        if (agg, 5) in curves:
            band(ax, curves[(agg, 5)], 'auc', AGG_LABEL[agg], AGG_COLOR[agg])
    ax.axvline(10, color='black', ls=':', alpha=0.6)
    ax.text(10.5, 0.52, 'Sybil nodes join (round 10)', fontsize=9)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Global AUC-ROC')
    ax.set_ylim(0.45, 0.95); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower left')
    fig.savefig(os.path.join(FIG, 'fig10_sybil_curves.png')); plt.close(fig)

    # Fig. 11 — AUC final vs fração de identidades Sybil
    fig, ax = plt.subplots(figsize=(7, 4))
    for agg in ['tbfl', 'fedavg', 'median', 'trimmed', 'krum']:
        s = syb[syb['agg'] == agg].sort_values('fraction')
        ax.errorbar(s['fraction'] * 100, s['auc_mean'], yerr=s['auc_sd'], marker='o',
                    capsize=3, color=AGG_COLOR[agg], label=AGG_LABEL[agg])
    ax.axvline(50, color='gray', ls=':')
    ax.set_xlabel('Sybil identities (% of participants)')
    ax.set_ylabel('AUC-ROC at round 50'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, 'fig11_sybil_fraction.png')); plt.close(fig)

    # Testes: TBFL vs cada baseline em S=5 e S=15 (por seed, teste t pareado)
    report['tests'] = {}
    for S in [5, 15]:
        t = load('tbfl', S, rounds=50)
        if t.empty:
            continue
        t = t[t['round'] == 50].set_index('seed')['auc']
        for agg in ['fedavg', 'median', 'trimmed', 'krum']:
            b = load(agg, S, rounds=50)
            if b.empty:
                continue
            b = b[b['round'] == 50].set_index('seed')['auc']
            common = t.index.intersection(b.index)
            diff = (t[common] - b[common]).values
            res = stats.ttest_rel(t[common], b[common]) if len(common) > 1 else None
            report['tests'][f'S{S}_tbfl_vs_{agg}'] = {
                'mean_diff_auc': round(float(diff.mean()), 4),
                't': None if res is None else round(float(res.statistic), 3),
                'p': None if res is None else float(res.pvalue), 'n_seeds': len(common)}

    meta = json.load(open(os.path.join(RES, 'tbfl_S0_seed0_meta.json')))
    report['dataset'] = {k: meta[k] for k in ['features', 'client_raw', 'client_pos_rate_raw',
                                             'client_sizes_balanced', 'test_n', 'test_pos',
                                             'deploy_gas']}
    json.dump(report, open(os.path.join(FIG, 'report.json'), 'w'), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    print(syb.round(4).to_string())


if __name__ == '__main__':
    main()
