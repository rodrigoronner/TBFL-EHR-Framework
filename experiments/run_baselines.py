"""
Baselines não federados (v2), com o mesmo MLP, otimizador e conjunto de features.

  --mode central_fed  treino centralizado nos mesmos dados de treino da federação
                      (união dos treinos locais já balanceados), avaliado no mesmo teste global
  --mode local        cada cliente treina sozinho; avaliado no teste local e no teste global
  --mode central_adm  split 80/20 global por internação + SMOTETomek global
  --mode central_pat  split 80/20 global por paciente (subject_id) + SMOTETomek global;
                      comparado a central_adm, mede o efeito de pacientes repetidos

Exemplo:
  python run_baselines.py --mode central_fed --seed 0 --epochs 30
  python run_baselines.py --all            # todos os modos, seeds 0-4, em paralelo
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from run_experiment import load_federation
from tbfl_data import NUMERIC, load_features
from tbfl_fl import MLP, get_flat, train_fedprox, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'data', 'mortalidade_features.csv')
SUBJECT_MAP = os.path.join(HERE, '..', 'data', 'hadm_subject_map.csv')
MODES = ['central_fed', 'local', 'central_adm', 'central_pat']


def fit(X, y, X_eval, y_eval, epochs, args, gen, every=1):
    """Treina o MLP do zero (mu=0, ou seja, SGD simples) e avalia a cada `every` épocas."""
    model = MLP(X.shape[1])
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    hist = []
    for ep in range(1, epochs + 1):
        train_fedprox(model, Xt, yt, get_flat(model), args, gen)
        if ep % every == 0 or ep == epochs:
            m = evaluate(model, X_eval, y_eval)
            m['epoch'] = ep
            hist.append(m)
    return model, hist


def global_split(group, seed, test_size=0.2):
    X, y, names = load_features(CSV)
    rng_state = seed
    if group == 'patient':
        hadm = pd.read_csv(CSV, usecols=['hadm_id'])['hadm_id']
        subj = hadm.map(pd.read_csv(SUBJECT_MAP).set_index('hadm_id')['subject_id']).values
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=rng_state)
                      .split(X, y, groups=subj))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y,
                                  random_state=rng_state)
    mu = X[tr][:, :len(NUMERIC)].mean(0)
    sd = X[tr][:, :len(NUMERIC)].std(0) + 1e-8
    X = X.copy()
    X[:, :len(NUMERIC)] = (X[:, :len(NUMERIC)] - mu) / sd
    X_bal, y_bal = SMOTETomek(random_state=seed).fit_resample(X[tr], y[tr])
    return X_bal.astype(np.float32), y_bal.astype(np.float32), X[te], y[te].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('TBFL_WORKERS', 8)))
    ap.add_argument('--cache', default=os.environ.get('TBFL_CACHE', os.path.join(HERE, '..', 'data', 'cache')))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'baselines'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.all:
        jobs = [(m, s) for m in MODES for s in range(5)]

        def run(job):
            m, s = job
            log = open(os.path.join(a.out, f'{m}_seed{s}.log'), 'w')
            rc = subprocess.call([sys.executable, __file__, '--mode', m, '--seed', str(s),
                                  '--epochs', str(a.epochs), '--cache', a.cache, '--out', a.out],
                                 stdout=log, stderr=subprocess.STDOUT)
            return f'{m}_seed{s}', rc

        with ThreadPoolExecutor(a.workers) as ex:
            for tag, rc in ex.map(run, jobs):
                print(f'{tag}: {"ok" if rc == 0 else "ERRO " + str(rc)}', flush=True)
        return

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    args = {'local_ep': 1, 'bs': a.bs, 'lr': a.lr, 'mu': 0.0}
    tag = f'{a.mode}_seed{a.seed}'

    if a.mode == 'central_fed':
        clients, X_test, y_test, _ = load_federation(CSV, 10, 0.5, a.seed, a.cache)
        X = np.concatenate([c['X_train'] for c in clients])
        y = np.concatenate([c['y_train'] for c in clients])
        _, hist = fit(X, y, X_test, y_test, a.epochs, args, gen)
        pd.DataFrame(hist).to_csv(os.path.join(a.out, f'{tag}.csv'), index=False)

    elif a.mode == 'local':
        clients, X_test, y_test, _ = load_federation(CSV, 10, 0.5, a.seed, a.cache)
        rows = []
        for k, c in enumerate(clients):
            model, hist = fit(c['X_train'], c['y_train'], c['X_test'], c['y_test'],
                              a.epochs, args, gen, every=a.epochs)
            g = evaluate(model, X_test, y_test)
            rows.append({'client': k, 'pos_rate': float(c['y_test'].mean()),
                         'local_auc': hist[-1]['auc'], 'local_auprc': hist[-1]['auprc'],
                         'global_auc': g['auc'], 'global_auprc': g['auprc'],
                         'global_recall': g['recall'], 'global_precision': g['precision']})
            print(f'[{tag}] client {k}: local AUC={hist[-1]["auc"]:.4f} '
                  f'global AUC={g["auc"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f'{tag}.csv'), index=False)

    else:
        X, y, X_te, y_te = global_split('patient' if a.mode == 'central_pat' else 'admission',
                                        a.seed)
        _, hist = fit(X, y, X_te, y_te, a.epochs, args, gen)
        d = pd.DataFrame(hist)
        d['test_n'], d['test_pos'] = len(y_te), int(y_te.sum())
        d.to_csv(os.path.join(a.out, f'{tag}.csv'), index=False)

    if a.mode != 'local':
        last = hist[-1]
        print(f'[{tag}] epoch {last["epoch"]}: AUC={last["auc"]:.4f} '
              f'AUPRC={last["auprc"]:.4f} Rec={last["recall"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
