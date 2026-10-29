# TBFL-EHR: Trustworthy Blockchain-based Federated Learning for Electronic Health Records

This repository holds the code, smart contracts and experiment logs for *"Trustworthy Blockchain-based Federated Learning for Electronic Health Records: Identity-Gated Participation against Sybil Attacks"* (Tertulino, Almeida and Alencar, IFRN; arXiv:2602.02629).

In TBFL, a hospital may contribute model updates to a federation only if it holds a credential issued by an accountable authority. The admission check runs in a smart-contract registry on an Ethereum-compatible ledger, before any update reaches the aggregator. The experiments compare this identity gate with unprotected FedAvg and with the Byzantine-robust aggregators coordinate-wise Median, Trimmed Mean and Krum, under coordinated Sybil attacks on MIMIC-IV in-hospital mortality prediction with 10 non-IID clients.

> **Version note.** Tag `v1.0-jsupercomputing` marks the code that accompanied the first manuscript version and arXiv:2602.02629v1. Several results reported in that version were produced by illustrative scripts rather than recorded runs. They are superseded by the experiments in `experiments/` and `results/`. Patient-level files derived from MIMIC-IV have been removed from the entire git history.

## Repository layout

```
contracts/      Solidity registries
  FLRegistry.sol          minimal issuer-controlled allowlist (used in all FL experiments)
  FLRegistryEvents.sol    same allowlist with indexed audit events
  FLRegistryZK.sol        EIP-712 signed credentials verified on-chain (expiry, revocation)
                          + Groth16 Merkle-membership path for anonymous submission
  MerkleMembershipVerifier.sol   snarkJS-generated verifier (GPL-3.0)
circuits/, zk/, scripts/zk/      Circom circuit and Groth16 toolchain (npm run zk:setup)
src/            DID/VC/ZK demo (did:ethr identities, EIP-712 credential issuance)
sql/            cohort definition used to build the dataset from MIMIC-IV
data/           empty: MIMIC-IV files are generated locally, never versioned
experiments/    federated learning, attacks, baselines, gas benchmark, figures
results/        per-run CSV logs (metrics per round, every on-chain transaction)
figures/        figures generated from results/
```

## Reproducing the experiments

**Requirements:**
- Python 3.13
- Node 22
- credentialed access to MIMIC-IV v3.1 loaded in PostgreSQL

**Steps:**

1. Install the dependencies.

   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   npm install
   ```

2. Build the dataset. See [data/README.md](data/README.md): run `sql/mortalidade_features.sql` and export the two CSV files into `data/`.

3. Start a local chain (200 accounts) in a separate terminal.

   ```bash
   npx hardhat node
   ```

4. Run the experiments. The full grid is 68 runs and takes about 6 hours on a 10-core CPU.

   ```bash
   cd experiments
   python run_all.py                         # Exp. 1 (no attack) + Exp. 2 (Sybil grid)
   python run_baselines.py --all             # centralized, local-only, patient-level split
   python bench_registry.py                  # gas/attack benchmark of the three registries
   python make_figures.py                    # figures/ + figures/report.json
   ```

A single run looks like this:

```bash
python run_experiment.py --agg tbfl --sybils 5 --seed 0 --rounds 50
# --agg {tbfl, fedavg, median, trimmed, krum}
```

## Experimental setup (summary)

| | |
|---|---|
| Data | MIMIC-IV v3.1: all 546,028 adult admissions, in-hospital mortality 2.16% |
| Features | 7 raw variables → 32 inputs (one-hot) |
| Federation | 10 clients, label-Dirichlet partition (α = 0.5), per-client 80/20 stratified split |
| Imbalance | SMOTETomek on each client's **training** split only; test sets keep the real prevalence |
| Model | MLP 32-64-32-1, FedProx (μ = 0.01), SGD (lr 0.01, momentum 0.9), E = 3, B = 32 |
| Attack | From round 10, S ∈ {2, 5, 10, 15} Sybils send the same N(0, I) vector and claim the mean client size |
| Ledger | Hardhat 2.22 (EVM, automine), Solidity 0.8.24, web3.py |

## Known limitations

- **Simulated federation.** The 10 clients are a statistical partition of one hospital's records, not independent institutions.
- **Repeated patients.** The unit is the admission (223,452 patients account for the 546,028 admissions), so a patient can appear in both training and test data. `run_baselines.py --mode central_pat` quantifies the effect with a patient-level split.
- **Discharge-time features.** `num_diagnoses` and `num_procedures` are only complete at discharge, so the task is retrospective risk stratification, not early warning.
- **Local chain only.** Latency was measured on a local automining node and excludes block production on a real network. Costs are extrapolated from measured gas.
- **Per-submission round counter.** In the registries, `currentTask.round` increments on every accepted submission, not once per FL round. The FL round is tracked off-chain. In `FLRegistryZK`, the ZK `externalNullifier` is therefore bound to that counter.
- **Insider attacks.** The identity gate does not defend against credentialed insiders that poison their own updates.

## License

MIT (see [LICENSE](LICENSE)), except for `contracts/MerkleMembershipVerifier.sol` (GPL-3.0, generated by snarkJS). MIMIC-IV is subject to the PhysioNet Credentialed Health Data Use Agreement.
