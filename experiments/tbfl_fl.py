"""
Núcleo de Federated Learning (v2): modelo, treino local FedProx,
regras de agregação e ataque Sybil.
"""
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss,
                             confusion_matrix)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)  # logits; sigmoid aplicada na avaliação


def get_flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def train_fedprox(model, X, y, global_flat, args, gen):
    """Treino local com termo proximal (mu/2)*||w - w_global||^2."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9,
                          weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    global_params = []
    i = 0
    for p in model.parameters():
        n = p.numel()
        global_params.append(global_flat[i:i + n].view_as(p))
        i += n
    n_samples = len(y)
    for _ in range(args['local_ep']):
        perm = torch.randperm(n_samples, generator=gen)
        for s in range(0, n_samples, args['bs']):
            b = perm[s:s + args['bs']]
            opt.zero_grad()
            loss = crit(model(X[b]).squeeze(1), y[b])
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            (loss + args['mu'] / 2 * prox).backward()
            opt.step()
    return get_flat(model)


# ------------------------- Agregadores -------------------------

def agg_fedavg(updates, weights):
    w = torch.tensor(weights, dtype=torch.float32)
    return (torch.stack(updates) * (w / w.sum()).unsqueeze(1)).sum(0)


def agg_median(updates, weights=None):
    return torch.stack(updates).median(0).values


def agg_trimmed_mean(updates, weights=None, beta=0.2):
    U = torch.stack(updates)
    n = U.shape[0]
    k = min(int(np.floor(beta * n)), (n - 1) // 2)
    S = U.sort(0).values
    return S[k:n - k].mean(0)


def agg_krum(updates, weights=None, f=1):
    U = torch.stack(updates)
    n = U.shape[0]
    f = min(f, max((n - 3) // 2, 0))  # Krum exige n > 2f + 2
    d = torch.cdist(U, U) ** 2
    m = max(n - f - 2, 1)
    scores = d.sort(1).values[:, 1:m + 1].sum(1)
    return U[scores.argmin()].clone()


def agg_foolsgold(updates, global_flat, history, ids):
    """
    FoolsGold (Fung et al., RAID 2020): pesos por participante a partir da similaridade
    de cosseno entre os históricos acumulados de atualizações (deltas), com "pardoning"
    e função logit. `history` persiste entre rodadas, indexado por `ids`.
    Devolve o novo modelo global e os pesos atribuídos.
    """
    deltas = [u - global_flat for u in updates]
    for i, d in zip(ids, deltas):
        history[i] = history[i] + d if i in history else d.clone()
    H = torch.stack([history[i] for i in ids]).double().numpy()
    n = len(ids)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    cs = Hn @ Hn.T - np.eye(n)
    maxcs = cs.max(1)
    for i in range(n):
        for j in range(n):
            if i != j and maxcs[i] < maxcs[j]:
                cs[i, j] *= maxcs[i] / maxcs[j]
    wv = np.clip(1 - cs.max(1), 0, 1)
    if wv.max() > 0:
        wv = wv / wv.max()
    wv[wv == 1] = 0.99
    with np.errstate(divide='ignore', invalid='ignore'):
        wv = np.log(wv / (1 - wv)) + 0.5
    wv = np.clip(np.nan_to_num(wv, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    if wv.sum() == 0:
        return global_flat.clone(), wv
    w = torch.tensor(wv, dtype=torch.float32)
    return global_flat + (torch.stack(deltas) * (w / w.sum()).unsqueeze(1)).sum(0), wv


# ------------------------- Ataque Sybil -------------------------

def sybil_updates(n_sybils, dim, gen, diversity=0.0):
    """
    Um único adversário controla todas as identidades Sybil. Com diversity=0, todas enviam
    o mesmo vetor N(0,1) (ataque coordenado). Com diversity>0 (ataque adaptativo), cada
    Sybil soma ao vetor comum um ruído próprio N(0, diversity^2), reduzindo a similaridade
    entre Sybils (cosseno esperado 1/(1+diversity^2)) para escapar de defesas por similaridade.
    """
    v = torch.randn(dim, generator=gen)
    if diversity <= 0:
        return [v.clone() for _ in range(n_sybils)]
    return [v + diversity * torch.randn(dim, generator=gen) for _ in range(n_sybils)]


# ------------------------- Avaliação -------------------------

@torch.no_grad()
def evaluate(model, X, y, threshold=0.5):
    model.eval()
    p = torch.sigmoid(model(torch.as_tensor(X)).squeeze(1)).numpy()
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    out = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'brier': brier_score_loss(y, p),
    }
    if len(np.unique(y)) > 1:
        out['auc'] = roc_auc_score(y, p)
        out['auprc'] = average_precision_score(y, p)
    else:
        out['auc'] = out['auprc'] = float('nan')
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               fnr=fn / max(fn + tp, 1))
    return out
