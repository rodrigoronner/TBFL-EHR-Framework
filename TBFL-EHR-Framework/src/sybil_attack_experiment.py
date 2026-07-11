"""
sybil_attack_experiment.py

Reproduces the Sybil-attack security experiment described in Sec. 5.4 of the paper
(Table 2): K legitimate clients train federated over several rounds; starting at a
fixed round, `N_SYBIL` malicious nodes join and submit pure Gaussian-noise model
updates. Two scenarios are compared:

  1. Baseline (Unprotected): the aggregator accepts every submitted update
     (standard FedAvg, no identity gate).
  2. TBFL (Proposed): the FLRegistry smart contract rejects updates from any
     address that was never issued a credential, so the Sybil updates never
     reach the aggregator.

Unlike the previous version of this experiment (which assumed a fixed ~20%
degradation and manufactured a synthetic baseline via `baseline = auc * 0.8 +
noise`), this script actually injects the attack and computes the reported
statistics (independent-samples t-test, Cohen's d) from real per-run
measurements across multiple random seeds.

The TBFL scenario requires a live local blockchain:
    Terminal A:  npx hardhat node
    Terminal B:  npx hardhat run scripts/deploy.js --network localhost
                 (then update CONTRACT_ADDRESS below with the printed address)
                 python src/sybil_attack_experiment.py

Because this trains K + N_SYBIL clients per round across N_RUNS x 2 scenarios,
ROUNDS/N_RUNS below default to a fast, illustrative configuration. Increase them
to reproduce the paper's full-scale 100-round numbers.
"""
import copy
import numpy as np
import pandas as pd
import torch
from scipy import stats

from data_loader import load_and_process_mimic, build_client_datasets
from cliente_fl import MLP, train_client_fedprox, MimicDataset
from main_tbfl_simulation import average_weights, evaluate_model, ARGS as BASE_ARGS
from blockchain_manager import BlockchainManager

CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
CSV_PATH = 'data/mortalidade_features.csv'

K_LEGIT = 10        # legitimate hospitals (paper Sec. 5.4)
N_SYBIL = 5         # Sybil nodes injected (33% of the post-injection network)
ATTACK_ROUND = 10   # round at which Sybil nodes start submitting
ROUNDS = 20          # total rounds simulated (>= ATTACK_ROUND); paper uses 100
N_RUNS = 5           # independent seeds per scenario, needed for a real t-test

LOCAL_ARGS = dict(BASE_ARGS)
LOCAL_ARGS['num_users'] = K_LEGIT


def run_scenario(seed, protected, X_train, y_train, X_test, y_test, user_groups):
    """
    Runs one full federated training with a Sybil attack injected at ATTACK_ROUND
    and returns the AUC-ROC time series on the held-out test set.

    protected=True  -> blockchain gate active: Sybil updates are rejected (TBFL).
    protected=False -> no gate: every submitted update is aggregated (baseline).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    client_arrays = build_client_datasets(X_train, y_train, user_groups)
    client_datasets = {cid: MimicDataset(X_c, y_c) for cid, (X_c, y_c) in client_arrays.items()}

    global_model = MLP(X_train.shape[1])

    bc = None
    workers = [f"legit_{i}" for i in range(K_LEGIT)]
    sybil_ids = [f"sybil_{i}" for i in range(N_SYBIL)]

    if protected:
        bc = BlockchainManager(CONTRACT_ADDRESS)
        workers = [bc.get_account(i + 1) for i in range(K_LEGIT)]
        # Sybil nodes reuse unused Hardhat accounts and are deliberately never authorized
        sybil_ids = [bc.get_account(K_LEGIT + 1 + i) for i in range(N_SYBIL)]
        for w in workers:
            bc.issue_credential(w)

    auc_history = []

    for round_idx in range(ROUNDS):
        local_weights, local_n = [], []

        # Legitimate clients always train and submit honestly
        for idx in range(K_LEGIT):
            w, _, n_k = train_client_fedprox(
                copy.deepcopy(global_model), client_datasets[idx], LOCAL_ARGS, global_model
            )
            accepted = bc.submit_hash(workers[idx], f"Qm_{round_idx}_{idx}")[0] if protected else True
            if accepted:
                local_weights.append(w)
                local_n.append(n_k)

        # Sybil nodes join from ATTACK_ROUND onward, submitting pure Gaussian noise
        if round_idx >= ATTACK_ROUND:
            avg_n = int(np.mean(local_n)) if local_n else 1
            for s_idx in range(N_SYBIL):
                noisy_state = {k: torch.randn_like(v) for k, v in global_model.state_dict().items()}
                accepted = bc.submit_hash(sybil_ids[s_idx], f"QmSybil_{round_idx}_{s_idx}")[0] if protected else True
                if accepted:
                    local_weights.append(noisy_state)
                    local_n.append(avg_n)

        if local_weights:
            global_model.load_state_dict(average_weights(local_weights, local_n))

        _, _, _, _, _, auc_val = evaluate_model(global_model, X_test, y_test)
        auc_history.append(auc_val)

    return auc_history


def main():
    print(f"🧪 Sybil Attack Experiment: K={K_LEGIT} legit clients, {N_SYBIL} Sybil nodes "
          f"injected at round {ATTACK_ROUND}, {N_RUNS} runs per scenario, {ROUNDS} rounds each.")

    X_train, y_train, X_test, y_test, user_groups = load_and_process_mimic(
        CSV_PATH, K_LEGIT, dirichlet_alpha=LOCAL_ARGS['dirichlet_alpha']
    )

    baseline_final_auc, tbfl_final_auc = [], []

    for run in range(N_RUNS):
        print(f"\n--- Run {run + 1}/{N_RUNS}: Baseline (Unprotected FedAvg) ---")
        auc_base = run_scenario(seed=run, protected=False, X_train=X_train, y_train=y_train,
                                 X_test=X_test, y_test=y_test, user_groups=user_groups)
        baseline_final_auc.append(auc_base[-1])
        print(f"   Final AUC: {auc_base[-1]:.4f}")

        print(f"--- Run {run + 1}/{N_RUNS}: TBFL (Blockchain-Protected) ---")
        try:
            auc_tbfl = run_scenario(seed=run, protected=True, X_train=X_train, y_train=y_train,
                                     X_test=X_test, y_test=y_test, user_groups=user_groups)
        except Exception as e:
            print(f"❌ Blockchain unavailable ({e}).")
            print("   Start 'npx hardhat node', deploy scripts/deploy.js, update "
                  "CONTRACT_ADDRESS above, then re-run this script.")
            return
        tbfl_final_auc.append(auc_tbfl[-1])
        print(f"   Final AUC: {auc_tbfl[-1]:.4f}")

    baseline_final_auc = np.array(baseline_final_auc)
    tbfl_final_auc = np.array(tbfl_final_auc)

    t_stat, p_val = stats.ttest_ind(tbfl_final_auc, baseline_final_auc, alternative='greater')
    pooled_std = np.sqrt((baseline_final_auc.var(ddof=1) + tbfl_final_auc.var(ddof=1)) / 2)
    cohens_d = (tbfl_final_auc.mean() - baseline_final_auc.mean()) / pooled_std if pooled_std > 0 else float('nan')

    print("\n📊 --- SYBIL ATTACK: TBFL vs UNPROTECTED BASELINE (real, measured) ---")
    print(f"   Baseline AUC (mean ± std, n={N_RUNS}): {baseline_final_auc.mean():.4f} ± {baseline_final_auc.std():.4f}")
    print(f"   TBFL AUC     (mean ± std, n={N_RUNS}): {tbfl_final_auc.mean():.4f} ± {tbfl_final_auc.std():.4f}")
    print(f"   T-Statistic: {t_stat:.4f} | P-Value: {p_val:.4e} | Cohen's d: {cohens_d:.2f}")

    pd.DataFrame({'run': range(N_RUNS), 'baseline_auc': baseline_final_auc, 'tbfl_auc': tbfl_final_auc}) \
        .to_csv('sybil_attack_results.csv', index=False)
    print("✅ Per-run results saved to sybil_attack_results.csv")


if __name__ == '__main__':
    main()
