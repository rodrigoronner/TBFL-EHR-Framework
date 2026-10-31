"""
Pipeline de dados corrigido (v2).

Ordem das etapas (sem vazamento):
  1. Particionamento não-IID entre K clientes (Dirichlet sobre o rótulo).
  2. Split estratificado 80/20 local em cada cliente.
  3. Padronização com estatísticas apenas dos conjuntos de treino.
  4. SMOTETomek aplicado somente ao treino local de cada cliente.
O conjunto de teste mantém a distribuição real (~2,2% de óbitos).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

NUMERIC = ['age', 'num_diagnoses', 'num_procedures']
TOP_CAREUNITS = 15


def _race_group(r):
    r = str(r).upper()
    if r.startswith('WHITE') or r == 'PORTUGUESE':
        return 'white'
    if r.startswith('BLACK'):
        return 'black'
    if 'HISPANIC' in r or r == 'SOUTH AMERICAN':
        return 'hispanic'
    if r.startswith('ASIAN'):
        return 'asian'
    if r in ('UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER'):
        return 'unknown'
    return 'other'


def load_features(csv_path):
    """Lê o CSV extraído do MIMIC-IV e devolve (X, y, nomes das features)."""
    df = pd.read_csv(csv_path)
    y = df['hospital_expire_flag'].astype(int).values

    cat = pd.DataFrame({
        'gender': (df['gender'] == 'M').astype(int),
    })
    race = df['race'].map(_race_group)
    insurance = df['insurance'].fillna('missing')
    top = df['first_careunit'].value_counts().index[:TOP_CAREUNITS]
    careunit = df['first_careunit'].where(df['first_careunit'].isin(top), 'other').fillna('other')

    onehot = pd.get_dummies(
        pd.DataFrame({'race': race, 'insurance': insurance, 'careunit': careunit}),
        dtype=int)
    X_df = pd.concat([df[NUMERIC].astype(float), cat, onehot], axis=1)
    return X_df.values.astype(np.float32), y, list(X_df.columns)


def dirichlet_partition(y, n_clients, alpha, rng, min_pos=50, min_size=2000):
    """Divide os índices entre clientes com proporções de classe ~ Dir(alpha)."""
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        sizes = [len(p) for p in parts]
        pos = [int(y[p].sum()) for p in parts]
        # Garante que todo cliente tenha amostras suficientes para split estratificado
        if min(sizes) >= min_size and min(pos) >= min_pos:
            return [np.array(p) for p in parts]


def build_federation(csv_path, n_clients=10, alpha=0.5, seed=0, test_size=0.2, val_frac=0.0):
    rng = np.random.default_rng(seed)
    X, y, names = load_features(csv_path)
    parts = dirichlet_partition(y, n_clients, alpha, rng)

    clients = []
    for k, idx in enumerate(parts):
        tr, te = train_test_split(idx, test_size=test_size, stratify=y[idx],
                                  random_state=seed + k)
        c = {'train_idx': tr, 'test_idx': te}
        if val_frac > 0:
            # Validação separada do treino antes do SMOTETomek, com a prevalência real
            # (usada apenas para escolher a rodada de parada; o teste não é consultado)
            c['train_idx'], c['val_idx'] = train_test_split(
                tr, test_size=val_frac, stratify=y[tr], random_state=seed + k + 100)
        clients.append(c)

    # Padronização com média/desvio agregados apenas dos treinos
    # (em produção, obtidos por agregação segura de estatísticas suficientes)
    all_train = np.concatenate([c['train_idx'] for c in clients])
    mu = X[all_train][:, :len(NUMERIC)].mean(0)
    sd = X[all_train][:, :len(NUMERIC)].std(0) + 1e-8
    X = X.copy()
    X[:, :len(NUMERIC)] = (X[:, :len(NUMERIC)] - mu) / sd

    for k, c in enumerate(clients):
        X_tr, y_tr = X[c['train_idx']], y[c['train_idx']]
        c['n_raw'] = len(y_tr)
        c['pos_rate_raw'] = float(y_tr.mean())
        smt = SMOTETomek(random_state=seed + k)
        X_bal, y_bal = smt.fit_resample(X_tr, y_tr)
        c['X_train'], c['y_train'] = X_bal.astype(np.float32), y_bal.astype(np.float32)
        c['X_test'], c['y_test'] = X[c['test_idx']], y[c['test_idx']].astype(np.float32)
        if 'val_idx' in c:
            c['X_val'], c['y_val'] = X[c['val_idx']], y[c['val_idx']].astype(np.float32)
            del c['val_idx']
        del c['train_idx'], c['test_idx']

    X_test = np.concatenate([c['X_test'] for c in clients])
    y_test = np.concatenate([c['y_test'] for c in clients])
    return clients, X_test, y_test, names
