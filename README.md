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

4. Run the experiments. The main grid is 68 runs (about 6 hours on a 10-core CPU) and the extra grid is 48 runs (about 5 hours).

   ```bash
   cd experiments
   python run_all.py                         # Exp. 1 (no attack) + Exp. 2 (Sybil grid)
   python run_all.py extra                   # Exp. 3: FoolsGold, adaptive Sybils, robust rules without attack
   python run_baselines.py --all             # centralized, local-only, patient-level split
   python bench_registry.py                  # gas/attack benchmark of the three registries
   python make_figures.py                    # figures/ + figures/report.json
   ```

A single run looks like this:

```bash
python run_experiment.py --agg tbfl --sybils 5 --seed 0 --rounds 50
# --agg {tbfl, fedavg, median, trimmed, krum, foolsgold}
# --sybil_diversity 1   adaptive attack: each Sybil adds its own N(0, I) noise
```

`results/` holds per-round metrics, on-chain transaction logs and run metadata for every run. Model checkpoints (`*.pt`) are not published because they are trained on MIMIC-IV. Without them, `make_figures.py` regenerates every panel except the ROC and precision–recall curves (`clin_b_roc`, `clin_c_pr`), which are included in `figures/` as generated locally.

## Public testnet deployments

The allowlist (`FLRegistry`) and signed-credential (`FLRegistryZK`) registries were deployed on two public test networks in October 2026. Every deployment, authorisation, admitted commitment, `ModelUpdated` event and revocation can be inspected on the block explorers. Rejected attack attempts revert during gas estimation and are never mined, so they appear only in `results/testnet_*.csv`.

| Network | Registry | Contract |
|---|---|---|
| Ethereum Sepolia | `FLRegistry` | [`0x54e698E73F26e18B17071BA471Cc6a11B6d5D57f`](https://sepolia.etherscan.io/address/0x54e698E73F26e18B17071BA471Cc6a11B6d5D57f) |
| Ethereum Sepolia | `FLRegistryZK` | [`0x2730b56bc97560D342a2032F6C4239c56eB5667F`](https://sepolia.etherscan.io/address/0x2730b56bc97560D342a2032F6C4239c56eB5667F) |
| Arbitrum Sepolia | `FLRegistry` | [`0xc4Ed367EEA1a699435F8351039b21e154b728476`](https://sepolia.arbiscan.io/address/0xc4Ed367EEA1a699435F8351039b21e154b728476) |
| Arbitrum Sepolia | `FLRegistryZK` | [`0x759F85E519Cb6ABcFCe14A4833ef8C071490857E`](https://sepolia.arbiscan.io/address/0x759F85E519Cb6ABcFCe14A4833ef8C071490857E) |

**Superseded deployments.** An earlier `FLRegistryZK` (Sepolia `0xCec575aF1b32b6311f36BF17dE60f87dEA843058`, Arbitrum Sepolia `0xC15a68e0605614Ff4a7C7EEb4f81274A170BBf11`) signed credentials over (subject, expiresAt) only, so anyone could re-relay a public credential after its revocation and re-authorise the revoked address. The current version signs an issuance time as well and accepts a credential only if it was issued after the last accepted credential and the last revocation of its subject; the benchmark now tests this case (`replayed_vc_after_revocation`). Do not use the superseded addresses.

To check that these contracts were created from the code in this repository, without relying on an explorer:

```bash
npx hardhat compile                      # reproduces the bytecode in experiments/*.json from contracts/
cd experiments && python verify_deployments.py
```

`verify_deployments.py` fetches each creation transaction from a public RPC endpoint and confirms that its data equal the released bytecode plus the constructor arguments, that it created the listed address and that it was sent by the issuer. To repeat the benchmark with your own funded key: `TBFL_TESTNET_KEY=... python testnet_bench.py --network sepolia` (or `arbsepolia`).

## Experimental setup (summary)

| | |
|---|---|
| Data | MIMIC-IV v3.1: all 546,028 adult admissions, in-hospital mortality 2.16% |
| Features | 7 raw variables → 32 inputs (one-hot) |
| Federation | 10 clients, label-Dirichlet partition (α = 0.5), per-client 80/20 stratified split |
| Imbalance | SMOTETomek on each client's **training** split only; test sets keep the real prevalence |
| Model | MLP 32-64-32-1, FedProx (μ = 0.01), SGD (lr 0.01, momentum 0.9), E = 3, B = 32 |
| Attack | From round 10, S ∈ {2, 5, 10, 15} Sybils send the same N(0, I) vector and claim the mean client size; adaptive variant adds per-Sybil noise (σ = 1) |
| Defences | FedAvg, coordinate-wise median, trimmed mean, Krum, FoolsGold, identity gate (TBFL) |
| Ledger | Hardhat 2.22 (EVM, automine), Solidity 0.8.24, web3.py |

## Known limitations

- **Simulated federation.** The 10 clients are a statistical partition of one hospital's records, not independent institutions.
- **Repeated patients.** The unit is the admission (223,452 patients account for the 546,028 admissions), so a patient can appear in both training and test data. `run_baselines.py --mode central_pat` quantifies the effect with a patient-level split.
- **Discharge-time features.** `num_diagnoses` and `num_procedures` are only complete at discharge, so the task is retrospective risk stratification, not early warning.
- **Local chain only.** Latency was measured on a local automining node and excludes block production on a real network. Costs are extrapolated from measured gas.
- **Per-submission round counter.** In the registries, `currentTask.round` increments on every accepted submission, not once per FL round. The FL round is tracked off-chain. In `FLRegistryZK`, the ZK `externalNullifier` is therefore bound to that counter.
- **Insider attacks.** The identity gate does not defend against credentialed insiders that poison their own updates.

## How to cite

The version that accompanies the PPNA manuscript is archived on Zenodo:

- **v2.1 (manuscript version):** [10.5281/zenodo.23836753](https://doi.org/10.5281/zenodo.23836753)
- **All versions (concept DOI, resolves to the latest release):** [10.5281/zenodo.23835125](https://doi.org/10.5281/zenodo.23835125)

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT (see [LICENSE](LICENSE)), except for `contracts/MerkleMembershipVerifier.sol` (GPL-3.0, generated by snarkJS). MIMIC-IV is subject to the PhysioNet Credentialed Health Data Use Agreement.
