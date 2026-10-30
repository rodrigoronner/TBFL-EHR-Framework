"""
Executa uma simulação federada (v2).

  --agg tbfl     FedAvg ponderado + verificação de identidade on-chain (proposto)
  --agg fedavg   FedAvg ponderado sem verificação (baseline desprotegido)
  --agg median | trimmed | krum   agregadores robustos sem verificação
  --agg foolsgold                 defesa anti-Sybil por similaridade (Fung et al., 2020)

Exemplo:
  python run_experiment.py --agg tbfl --sybils 5 --seed 0 --rounds 100
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from tbfl_data import build_federation
from tbfl_fl import (MLP, get_flat, set_flat, train_fedprox, evaluate, sybil_updates,
                     agg_fedavg, agg_median, agg_trimmed_mean, agg_krum, agg_foolsgold)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_federation(csv, n_clients, alpha, seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'fed_k{n_clients}_a{alpha}_s{seed}.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    fed = build_federation(csv, n_clients, alpha, seed)
    pickle.dump(fed, open(path, 'wb'))
    return fed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg', default='tbfl',
                    choices=['tbfl', 'fedavg', 'median', 'trimmed', 'krum', 'foolsgold'])
    ap.add_argument('--sybils', type=int, default=0)
    ap.add_argument('--attack_start', type=int, default=10)
    ap.add_argument('--sybil_diversity', type=float, default=0.0,
                    help='0 = Sybils idênticos; >0 = ataque adaptativo (ruído próprio por Sybil)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--clients', type=int, default=10)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--local_ep', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--mu', type=float, default=0.01)
    ap.add_argument('--per_client', action='store_true',
                    help='avalia o modelo global no teste local de cada cliente')
    ap.add_argument('--account_offset', type=int, default=0)
    ap.add_argument('--csv', default=os.path.join(HERE, '..', 'data', 'mortalidade_features.csv'))
    ap.add_argument('--cache', default=os.environ.get('TBFL_CACHE', os.path.join(HERE, '..', 'data', 'cache')))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results'))
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    args = {'local_ep': a.local_ep, 'bs': a.bs, 'lr': a.lr, 'mu': a.mu}

    clients, X_test, y_test, names = load_federation(a.csv, a.clients, a.alpha, a.seed, a.cache)
    for c in clients:
        c['Xt'] = torch.as_tensor(c['X_train'])
        c['yt'] = torch.as_tensor(c['y_train'])
    n_honest = len(clients)
    sizes = [len(c['y_train']) for c in clients]
    claimed = int(np.mean(sizes))  # Sybils declaram o tamanho médio de um cliente honesto

    model = MLP(len(names))
    global_flat = get_flat(model)
    dim = global_flat.numel()

    chain = None
    honest_addr, sybil_addr = [], []
    if a.agg == 'tbfl':
        from tbfl_chain import Chain
        chain = Chain(account_offset=a.account_offset)
        honest_addr = chain.pool[:n_honest]
        sybil_addr = chain.pool[n_honest:n_honest + a.sybils]
        for addr in honest_addr:
            chain.authorize(addr, 0)

    div = f'_div{a.sybil_diversity:g}' if a.sybil_diversity > 0 else ''
    tag = f'{a.agg}_S{a.sybils}{div}_seed{a.seed}'
    fg_history = {}
    os.makedirs(a.out, exist_ok=True)
    history, client_hist = [], []

    for r in range(1, a.rounds + 1):
        updates, weights, ids, train_t, chain_t = [], [], [], [], []
        n_mismatch = 0
        for k, c in enumerate(clients):
            local = MLP(len(names))
            set_flat(local, global_flat)
            t0 = time.perf_counter()
            w = train_fedprox(local, c['Xt'], c['yt'], global_flat, args, gen)
            train_t.append(time.perf_counter() - t0)
            if chain is not None:
                t1 = time.perf_counter()
                rcpt = chain.submit(honest_addr[k], w, r)
                chain_t.append(time.perf_counter() - t1)
                if rcpt is None:
                    continue
                # O agregador só usa a atualização se ela corresponder ao compromisso admitido
                if not chain.verify(rcpt, honest_addr[k], w):
                    n_mismatch += 1
                    continue
            updates.append(w)
            weights.append(sizes[k])
            ids.append(k)

        attacking = a.sybils > 0 and r >= a.attack_start
        n_rejected = 0
        if attacking:
            syb = sybil_updates(a.sybils, dim, gen, a.sybil_diversity)
            for j, v in enumerate(syb):
                if chain is not None:
                    if r == a.attack_start:
                        chain.self_authorize(sybil_addr[j], r)
                    rcpt = chain.submit(sybil_addr[j], v, r, kind='sybil_submit')
                    if rcpt is None or not chain.verify(rcpt, sybil_addr[j], v):
                        n_rejected += 1
                        continue
                updates.append(v)
                weights.append(claimed)
                ids.append(n_honest + j)

        if a.agg in ('tbfl', 'fedavg'):
            global_flat = agg_fedavg(updates, weights)
        elif a.agg == 'median':
            global_flat = agg_median(updates)
        elif a.agg == 'trimmed':
            beta = min(max(a.sybils, 1) / len(updates), 0.49) if attacking else 0.1
            global_flat = agg_trimmed_mean(updates, beta=beta)
        elif a.agg == 'krum':
            global_flat = agg_krum(updates, f=a.sybils if attacking else 1)
        elif a.agg == 'foolsgold':
            global_flat, fg_w = agg_foolsgold(updates, global_flat, fg_history, ids)

        # Fração do peso de agregação efetivamente atribuída a identidades Sybil
        if a.agg == 'foolsgold':
            wts = np.asarray(fg_w, dtype=float)
        elif a.agg in ('tbfl', 'fedavg'):
            wts = np.asarray(weights, dtype=float)
        else:
            wts = None
        sybil_share = (float(wts[[i >= n_honest for i in ids]].sum() / wts.sum())
                       if wts is not None and wts.sum() > 0 else float('nan'))

        set_flat(model, global_flat)
        m = evaluate(model, X_test, y_test)
        m.update(round=r, n_aggregated=len(updates), n_rejected=n_rejected,
                 n_hash_mismatch=n_mismatch,
                 sybil_weight_share=sybil_share,
                 avg_train_time=float(np.mean(train_t)),
                 avg_chain_time=float(np.mean(chain_t)) if chain_t else 0.0)
        history.append(m)

        if a.per_client and (r % 5 == 0 or r == 1):
            for k, c in enumerate(clients):
                mc = evaluate(model, c['X_test'], c['y_test'])
                client_hist.append({'round': r, 'client': k, 'auc': mc['auc'],
                                    'auprc': mc['auprc'], 'recall': mc['recall'],
                                    'pos_rate': float(c['y_test'].mean()),
                                    'n_test': len(c['y_test'])})
        if r % 10 == 0:
            print(f'[{tag}] R{r}: AUC={m["auc"]:.4f} AUPRC={m["auprc"]:.4f} '
                  f'Rec={m["recall"]:.3f} Prec={m["precision"]:.3f} agg={len(updates)}',
                  flush=True)

    pd.DataFrame(history).to_csv(os.path.join(a.out, f'{tag}.csv'), index=False)
    if client_hist:
        pd.DataFrame(client_hist).to_csv(os.path.join(a.out, f'{tag}_clients.csv'), index=False)
    if chain is not None:
        pd.DataFrame(chain.log).to_csv(os.path.join(a.out, f'{tag}_chain.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(a.out, f'{tag}_model.pt'))
    meta = vars(a) | {'features': names, 'client_sizes_balanced': sizes,
                      'client_raw': [c['n_raw'] for c in clients],
                      'client_pos_rate_raw': [c['pos_rate_raw'] for c in clients],
                      'test_n': int(len(y_test)), 'test_pos': int(y_test.sum()),
                      'deploy_gas': chain.deploy_gas if chain else None}
    json.dump(meta, open(os.path.join(a.out, f'{tag}_meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
