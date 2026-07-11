import torch
import torch.nn as nn
import copy
import numpy as np
import time
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import load_and_process_mimic, build_client_datasets
from cliente_fl import MLP, train_client_fedprox, MimicDataset
from blockchain_manager import BlockchainManager

# ================= CONFIGURATIONS =================
# Address of the deployed Smart Contract on the local Hardhat network
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
CSV_PATH = 'data/mortalidade_features.csv'

ARGS = {
    'rounds': 100,           # Number of global training rounds (Long-term simulation)
    'num_users': 10,         # K=10 participating hospitals (paper Sec. 3.1.2 / 5.4)
    'local_ep': 3,           # Local epochs per global round
    'bs': 32,                # Batch size
    'lr': 0.01,              # Learning rate (paper: SGD, lr=0.01)
    'momentum': 0.9,         # SGD momentum (paper Sec. 3.1.3)
    'weight_decay': 1e-5,    # SGD weight decay (paper Sec. 3.1.3)
    'mu': 0.01,              # FedProx proximal term coefficient
    'dirichlet_alpha': 0.5,  # Non-IID heterogeneity concentration (paper Sec. 3.1.2)
}
# =================================================

def average_weights(w_list, n_samples):
    """
    Performs weighted Federated Averaging (FedAvg), weighting each client's
    contribution by its local training-set size n_k, per Algorithm 2 of the paper:
        w_t = sum_k (n_k / sum_j n_j) * w_k

    Args:
        w_list (list): List of state_dicts from authorized clients.
        n_samples (list): Number of local training samples for each entry in w_list.

    Returns:
        state_dict: The weighted-averaged global model weights.
    """
    total_samples = sum(n_samples)
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        w_avg[key] = w_list[0][key] * (n_samples[0] / total_samples)
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key] * (n_samples[i] / total_samples)
    return w_avg

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the global model on the hold-out test set.

    Returns:
        tuple: (avg_loss, accuracy, precision, recall, f1_score, auc_roc)
    """
    model.eval()
    criterion = nn.BCELoss() # Binary Cross Entropy
    dataset = MimicDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true = []
    y_pred_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            y_pred_probs.extend(outputs.numpy())
            y_true.extend(labels.numpy())

    # Calculate averages
    avg_loss = total_loss / len(loader)
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_cls = (y_pred_probs > 0.5).astype(int)

    # Scikit-Learn Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred_cls)
    prec = precision_score(y_true, y_pred_cls, zero_division=0)
    rec = recall_score(y_true, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true, y_pred_cls, zero_division=0)
    try:
        auc_val = roc_auc_score(y_true, y_pred_probs)
    except:
        auc_val = 0.5 # Fallback if only one class is present

    return avg_loss, acc, prec, rec, f1, auc_val

def analyze_statistics(df_history):
    """
    Detects the round at which global accuracy stabilizes, using a moving
    standard deviation (window=5) that must drop below 0.002.

    Note: the comparative Sybil-attack security analysis (TBFL vs. an unprotected
    baseline) is a separate controlled experiment - see sybil_attack_experiment.py,
    which actually injects malicious nodes rather than assuming a fixed degradation.
    """
    print("\n📊 --- AUTOMATED STATISTICAL ANALYSIS ---")

    acc_series = df_history['accuracy'].values

    window = 5
    stabilization_point = ARGS['rounds'] # Default: did not stabilize
    rolling_std = pd.Series(acc_series).rolling(window=window).std()

    stable_indices = np.where(rolling_std < 0.002)[0]
    if len(stable_indices) > 0:
        stabilization_point = stable_indices[0]

    print(f"🔹 Estimated Stabilization Point: Round {stabilization_point}")
    print(f"🔹 Final Accuracy: {acc_series[-1]:.4f}")
    print("ℹ️  For the Sybil-attack security comparison (Table 2 / Sec. 5.4), run "
          "sybil_attack_experiment.py, which simulates the actual attack instead of "
          "assuming a fixed degradation.")

    return stabilization_point

def main():
    print(f"🚀 Starting Real TBFL Simulation ({ARGS['rounds']} Rounds, K={ARGS['num_users']} clients)...")

    # 1. Initialize Blockchain Manager
    try:
        bc = BlockchainManager(CONTRACT_ADDRESS)
    except Exception as e:
        print(f"Blockchain Error: {e}")
        return

    # 2. Load Data (global 80/20 split, then Dirichlet(alpha) non-IID partitioning)
    X_train, y_train, X_test, y_test, user_groups = load_and_process_mimic(
        CSV_PATH, ARGS['num_users'], dirichlet_alpha=ARGS['dirichlet_alpha']
    )

    # 3. Build per-client datasets. SMOTETomek is applied exclusively to each
    #    client's local training fold; the global test set stays untouched (Sec. 3.1.4).
    print("⚖️  Applying SMOTETomek balancing to each client's local training fold...")
    client_arrays = build_client_datasets(X_train, y_train, user_groups)
    client_datasets = {cid: MimicDataset(X_c, y_c) for cid, (X_c, y_c) in client_arrays.items()}

    # 4. Initialize Global Model
    input_dim = X_train.shape[1]
    global_model = MLP(input_dim)
    global_model.train()
    global_weights = global_model.state_dict()

    # 5. Identity Management: all K hospitals are legitimate and receive credentials.
    #    (Unauthorized/Sybil nodes are evaluated separately in sybil_attack_experiment.py)
    workers = [bc.get_account(i + 1) for i in range(ARGS['num_users'])]
    for worker_addr in workers:
        bc.issue_credential(worker_addr)

    history = []

    # 6. FL Training Loop
    for round_idx in range(ARGS['rounds']):

        local_weights = []
        local_n_samples = []
        blockchain_times = []
        training_times = []

        # Iterate over clients
        for idx in range(ARGS['num_users']):
            worker_addr = workers[idx]

            # Local Training
            t0 = time.time()
            w, _, n_k = train_client_fedprox(
                copy.deepcopy(global_model), client_datasets[idx], ARGS, global_model
            )
            training_times.append(time.time() - t0)

            # Blockchain Verification
            t0_bc = time.time()
            # Simulate IPFS Hash (in production, this would be a real CID)
            fake_ipfs = f"QmHash_{round_idx}_{worker_addr[:5]}"

            # Attempt to submit to Smart Contract
            accepted, _ = bc.submit_hash(worker_addr, fake_ipfs)
            blockchain_times.append(time.time() - t0_bc)

            # Aggregation Logic: Only include weights if Blockchain accepted the submission
            if accepted:
                local_weights.append(copy.deepcopy(w))
                local_n_samples.append(n_k)

        # Global Aggregation (weighted by each client's local sample count)
        if len(local_weights) > 0:
            global_weights = average_weights(local_weights, local_n_samples)
            global_model.load_state_dict(global_weights)

            # Evaluation
            loss, acc, prec, rec, f1, auc_val = evaluate_model(global_model, X_test, y_test)

            if (round_idx + 1) % 10 == 0:
                print(f"\n   📅 R{round_idx+1}: Loss={loss:.4f} | Acc={acc:.4f} | AUC={auc_val:.4f}")

            history.append({
                'round': round_idx + 1,
                'loss': loss,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc_val,
                'avg_train_time': np.mean(training_times),
                'avg_blockchain_time': np.mean(blockchain_times)
            })

    # Save Results
    df_res = pd.DataFrame(history)
    df_res.to_csv('tbfl_simulation_results.csv', index=False)
    print("\n✅ Simulation complete. Results saved to CSV.")

    # Statistical Analysis
    analyze_statistics(df_res)

if __name__ == '__main__':
    main()
