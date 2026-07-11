# TBFL-EHR: Trustworthy Blockchain-based Federated Learning for Electronic Health Records

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardhat](https://img.shields.io/badge/built%20with-Hardhat-FFDB1C.svg)](https://hardhat.org/)
[![DOI](https://img.shields.io/badge/https://arxiv.org/abs/2602.02629-blue)](https://arxiv.org/abs/2602.02629)

> Official implementation of **"Trustworthy Blockchain-based Federated Learning for Electronic Health Records: Securing Participant Identity with Decentralized Identifiers and Verifiable Credentials"**

## 📖 Overview

This repository implements a novel secure Federated Learning framework that combines **Self-Sovereign Identity (SSI)** principles with **Blockchain technology** to enable privacy-preserving collaborative machine learning across healthcare institutions. By leveraging Decentralized Identifiers (DIDs) and Verifiable Credentials (VCs), our architecture ensures that only authenticated and authorized healthcare entities can participate in model training.

### Key Features

- ✅ **100% Sybil Attack Prevention**: Cryptographic identity verification eliminates unauthorized participation
- ✅ **Robust Clinical Performance**: AUC = 0.954, Recall = 0.890 on MIMIC-IV mortality prediction
- ✅ **Minimal Overhead**: <0.12% computational latency from blockchain verification
- ✅ **Economic Viability**: ~$18 total cost for 100 training rounds across multiple institutions
- ✅ **Privacy-Preserving**: Compliant with GDPR and HIPAA regulations

---

## 🏗️ System Architecture

The TBFL framework consists of three synergistic layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐         ┌──────────────┐│
│  │  Hospital 1  │  │  Hospital 2  │  . . .  │ Hospital K=10││
│  │   (Client)   │  │   (Client)   │         │   (Client)   ││
│  └──────┬───────┘  └──────┬───────┘         └──────┬───────┘│
│         │                  │                        │        │
│         └──────────────────┴────────────────────────┘        │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│              BLOCKCHAIN LAYER (Identity Verification)        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FLRegistry.sol (address allowlist, used by the      │   │
│  │  main FL simulation) or FLRegistryZK.sol (DID/VC via  │   │
│  │  EIP-712 + Groth16 ZK selective disclosure)           │   │
│  │  • Authorization Registry                            │   │
│  │  • Model Hash Recording                              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┼─────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│           FEDERATED LEARNING LAYER (Model Training)          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FedProx Algorithm + SMOTETomek Balancing            │   │
│  │  • Local Training (MLP)                              │   │
│  │  • Secure Aggregation                                │   │
│  │  • Convergence Monitoring                            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

1. **Blockchain Layer (Ethereum/Hardhat)**
   - The main FL simulation gates participation via `FLRegistry.sol`, a lightweight
     address allowlist managed by a Trusted Issuer
   - `FLRegistryZK.sol` implements the full DID/Verifiable Credential/Zero-Knowledge
     identity layer described in Sec. 4.1 — see *Identity Layer* below
   - Records an immutable audit trail of model updates
   - **Gas Cost**: ~0.18 USD per round with linear scalability

2. **Federated Learning Layer (PyTorch)**
   - Implements FedProx optimization algorithm for non-IID data
   - Applies SMOTETomek balancing to address class imbalance
   - Performs secure weighted averaging of validated model updates
   - **Training Time**: ~17 seconds per round (local), ~0.02s blockchain verification

3. **Data Processing Pipeline (MIMIC-IV)**
   - SQL-based cohort selection from a PostgreSQL instance
   - Python-based feature engineering and preprocessing
   - Automated tensor conversion for distributed training
   - **Dataset Size**: 546,028 ICU admissions after filtering

---

## 📋 Prerequisites

### Required Software

- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher (with NPM)
- **Git**: Version control

### Required Access

- **MIMIC-IV Database**: Credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
  - Complete CITI "Data or Specimens Only Research" training
  - Sign Data Use Agreement (DUA)
  - Download MIMIC-IV v2.2 or v3.1

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/rodrigoronner/TBFL-EHR-Framework.git
cd TBFL-EHR-Framework
```

### 2. Blockchain Environment Setup

Install Hardhat and required Ethereum packages:

```bash
npm install
```

**Key Dependencies** (auto-installed):
- `hardhat`: Ethereum development environment
- `@nomiclabs/hardhat-ethers`: Ethereum library integration
- `@nomiclabs/hardhat-waffle`: Testing framework
- `chai`: Assertion library for tests

### 3. Python Environment Setup

Create isolated virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

**Key Dependencies** (from `requirements.txt`):
- `torch>=2.0.0`: Deep learning framework
- `web3`: Ethereum blockchain interaction
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `imbalanced-learn`: SMOTETomek implementation
- `scipy`: Statistical analysis (Sybil-attack significance testing)

---

## 🧪 Experimental Replication Guide

### Step 1: Prepare MIMIC-IV Dataset

**Option A: Use Provided Preprocessed Dataset (Recommended)**

The repository includes a preprocessed, de-identified dataset that meets all inclusion criteria:

```bash
# Extract the provided dataset
cd data/
unzip mortalidade_features.csv.zip
cd ..
# This creates mortalidade_features.csv (546,028 admissions)
```

The dataset is automatically loaded by `main_tbfl_simulation.py` during execution.

**Option B: Generate Dataset from Raw MIMIC-IV**

If you have access to MIMIC-IV and wish to reproduce the preprocessing:

1. Access MIMIC-IV database (version 2.2 or 3.1) via PhysioNet
2. Write a cohort-selection query against your own PostgreSQL instance (the repository
   does not currently ship the SQL used to build `mimiciv_hosp.mortalidade_features`;
   this is tracked as a reproducibility gap — see the Contributing section)
3. Apply inclusion criteria:
   - Adult patients (age ≥ 18 years)
   - First admission only (prevent data leakage)
   - Non-null mortality outcome (`hospital_expire_flag`)
4. Save result as `data/mortalidade_features.csv`

**Dataset Validation**:
```bash
# Verify dataset was extracted correctly
python -c "import pandas as pd; df = pd.read_csv('data/mortalidade_features.csv'); print(f'Loaded {len(df)} admissions')"
# Expected output: Loaded 546028 admissions
```

### Step 2: Launch Local Blockchain Node

Open **Terminal A** (keep running throughout experiment):

```bash
npx hardhat node
```

**Expected Output**:
```
Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/

Accounts
========
Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
Account #1: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 (10000 ETH)
...
```

### Step 3: Deploy Smart Contract

Open **Terminal B**:

```bash
npx hardhat run scripts/deploy.js --network localhost
```

**Critical Step** - Copy the contract address from output:
```
FLRegistry deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
```

Update `src/main_tbfl_simulation.py`:
```python
# Line 15-20
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'  # ← Paste here
```

### Step 4: Execute Federated Learning Simulation

In **Terminal B** (with activated Python environment):

```bash
python src/main_tbfl_simulation.py
```

**Execution Flow** (100 rounds, K=10 hospitals per round):

```
🚀 Starting Real TBFL Simulation (100 Rounds, K=10 clients)...
🔗 Blockchain Connected. Contract loaded at: 0x5FbDB2315678afecb367f032d93F642f64180aa3
📂 Loading data from: data/mortalidade_features.csv
✅ Data Processed. Shape: (546028, 85)
⚖️  Applying SMOTETomek balancing to each client's local training fold...
🏛️  Trusted Issuer issuing credential to: 0x709979... (repeats for all 10 hospitals)
   ✅ Credential successfully registered on the ledger.

   📅 R10: Loss=0.2306 | Acc=0.9082 | AUC=0.8534
   📅 R20: Loss=0.2318 | Acc=0.9114 | AUC=0.8977
   ...
   📅 R100: Loss=0.2724 | Acc=0.8810 | AUC=0.9540
✅ Simulation complete. Results saved to CSV.
```

### Step 5: Analyze Results

The simulation outputs performance metrics directly to the console for each round:
- **Per-round metrics**: Loss, Accuracy, AUC-ROC
- **Blockchain metrics**: Gas consumption, transaction confirmation time
- **Security validation**: Authorization checks, Sybil attack prevention

**Expected Final Results (Round 100)**:
- Global Accuracy: ~88.1%
- Global AUC-ROC: ~0.954
- Global Recall: ~0.890
- Average Gas per round: ~45,200
- Blockchain overhead: <0.12%

---

## 📊 Expected Results

### Performance Metrics (Final Global Model)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **AUC-ROC** | 0.954 | Excellent discriminative ability |
| **Recall (Sensitivity)** | 0.890 | Captures 89% of mortality cases |
| **Precision** | 0.876 | High confidence in positive predictions |
| **F1-Score** | 0.883 | Balanced performance |
| **Accuracy** | 0.881 | Overall correct classifications |

### Security Validation

| Attack Type | Attempts | Blocked | Success Rate |
|-------------|----------|---------|--------------|
| **Sybil Attack** | 500 | 500 | **100%** |
| **Unauthorized Access** | 350 | 350 | **100%** |
| **Credential Forgery** | 200 | 200 | **100%** |

### Operational Efficiency

| Component | Metric | Value |
|-----------|--------|-------|
| **Local Training** | Time per round | ~17.0 seconds |
| **Blockchain Verification** | Time per round | ~0.02 seconds |
| **Overhead** | Percentage | **0.12%** |
| **Gas Cost** | Per round | ~0.18 USD |
| **Total Cost** | 100 rounds | ~18 USD |

---

## 🗂️ Repository Structure

```
TBFL-EHR-Framework/
│
├── contracts/                  # Ethereum Smart Contracts
│   ├── FLRegistry.sol         # Main access control contract (used by the FL simulation)
│   ├── FLRegistryZK.sol       # DID/Verifiable Credential/ZK identity layer (Sec. 4.1)
│   └── MerkleMembershipVerifier.sol  # snarkjs-generated Groth16 verifier
│
├── circuits/                   # Zero-Knowledge circuit source
│   └── merkleMembership.circom
│
├── scripts/                    # Deployment and utilities
│   ├── deploy.js              # FLRegistry deployment script
│   ├── deploy_zk.js           # FLRegistryZK + verifier deployment script
│   └── zk/                    # Node/snarkjs bridge (proof generation, trusted setup)
│
├── src/                        # Federated Learning Core
│   ├── main_tbfl_simulation.py      # Main execution script (K=10, 100 rounds)
│   ├── sybil_attack_experiment.py   # Real Sybil-attack security experiment (Sec. 5.4)
│   ├── blockchain_manager.py        # Web3 interface
│   ├── cliente_fl.py                # FL client (hospital): MLP + FedProx
│   ├── data_loader.py               # Dirichlet partitioning + per-client SMOTETomek
│   ├── identity.py                  # DID (did:ethr) + EIP-712 Verifiable Credentials
│   ├── zk_bridge.py                 # Python <-> Node bridge for ZK proof generation
│   ├── demo_did_vc_zk.py            # Standalone DID/VC/ZK walkthrough demo
│   └── demo_security_mechanism.py   # Standalone FLRegistry.sol access-control demo
│
├── zk/keys/verification_key.json  # Groth16 verification key (zkey/ptau are gitignored)
│
├── data/                       # Data directory
│   ├── mortalidade_features.csv.zip # Compressed preprocessed dataset
│   ├── mortalidade_features.csv     # Extracted dataset (after unzip)
│   └── raw/                         # Raw MIMIC-IV files (optional)
│
├── hardhat.config.js           # Hardhat configuration
├── package.json                # Node.js dependencies
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                   # This file
```

### Directory Descriptions

- **`contracts/`**: Solidity smart contracts for blockchain-based access control
  - `FLRegistry.sol`: Address-allowlist access control used by the FL simulation
  - `FLRegistryZK.sol`: DID + EIP-712 Verifiable Credentials + Groth16 ZK selective disclosure
  - `MerkleMembershipVerifier.sol`: Generated Groth16 verifier for the ZK circuit

- **`circuits/`**: `merkleMembership.circom`, compiled and put through a Groth16 trusted
  setup by `npm run zk:setup` (see *Identity Layer*)

- **`scripts/`**: Deployment automation and utility scripts
  - `deploy.js` / `deploy_zk.js`: Deploy `FLRegistry` / `FLRegistryZK` + verifier
  - `zk/`: Node.js helpers (Merkle tree, commitment, proof generation) called from Python via `zk_bridge.py`

- **`src/`**: Core federated learning implementation
  - `main_tbfl_simulation.py`: Orchestrates the complete TBFL workflow (K=10, 100 rounds)
  - `sybil_attack_experiment.py`: Injects real Sybil nodes mid-training and compares an
    unprotected baseline against the blockchain-gated scenario
  - `blockchain_manager.py`: Web3.py interface for smart contract interaction
  - `cliente_fl.py`: Hospital client implementing MLP training and FedProx optimization
  - `data_loader.py`: Dirichlet(α) non-IID partitioning and per-client SMOTETomek balancing
  - `identity.py` / `zk_bridge.py` / `demo_did_vc_zk.py`: DID/VC/ZK identity layer

- **`data/`**: Dataset directory (preprocessed data included)
  - `mortalidade_features.csv.zip`: Compressed preprocessed MIMIC-IV cohort (included in repository)
  - `mortalidade_features.csv`: Extracted dataset with 546,028 ICU admissions (created after unzip)
  - `raw/`: Placeholder for raw MIMIC-IV files (not distributed due to DUA)
  
  **Note**: The main script automatically loads `mortalidade_features.csv` during execution

- **Configuration Files**:
  - `hardhat.config.js`: Ethereum network configuration (localhost, testnets)
  - `package.json`: Node.js dependencies (Hardhat, Ethers.js)
  - `requirements.txt`: Python dependencies (PyTorch, Web3, Scikit-learn)
  - `.env.example`: Template for environment variables (private keys, database credentials)

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Blockchain Configuration
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
INFURA_API_KEY=your_infura_key_here  # For mainnet deployment

# Simulation Parameters
NUM_CLIENTS=10
NUM_ROUNDS=100
LEARNING_RATE=0.01
FEDPROX_MU=0.01
```

**Note**: these simulation parameters are currently hardcoded in the `ARGS` dict at the
top of `src/main_tbfl_simulation.py` rather than read from `.env` — edit that dict directly
to change them for now.

### Hardhat Network Configuration

The current `hardhat.config.js` is intentionally minimal:

```javascript
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.24",
};
```

`npx hardhat node` defaults to chain ID `31337`. To deploy to a testnet, add a `networks`
block, for example:

```javascript
module.exports = {
  solidity: "0.8.24",
  networks: {
    sepolia: {
      url: `https://sepolia.infura.io/v3/${INFURA_API_KEY}`,
      accounts: [PRIVATE_KEY]
    }
  }
};
```

---

## 🧩 Component Details

### Smart Contract: `FLRegistry.sol`

**Key Functions**:

```solidity
// Trusted Issuer authorizes a hospital to participate
function authorizeWorker(address worker) external

// Submit model hash after local training (reverts if not authorized)
function submitUpdate(string memory ipfsHash) external

// Query authorization status
function authorizedWorkers(address worker) external view returns (bool)
```

**Events**:
```solidity
event WorkerAuthorized(address indexed worker);
event ModelUpdated(uint round, string newHash, address indexed contributor);
```

For the DID/Verifiable Credential/Zero-Knowledge version of this contract, see
`FLRegistryZK.sol` under *Identity Layer*.

### Python Modules

**`blockchain_manager.py`**: Web3 interface
- ABI loading and contract interaction
- Transaction signing via the connected node's own accounts
- Gas estimation and monitoring
- Connection to local Hardhat node

**`data_loader.py`**: Data pipeline
- Loads and preprocesses the MIMIC-derived CSV (imputation, one-hot encoding, scaling)
- Dirichlet(α=0.5) non-IID partitioning across clients (`partition_data_dirichlet`)
- Per-client SMOTETomek balancing, applied only to local training folds (`build_client_datasets`)

**`cliente_fl.py`**: Federated Learning client
- MLP architecture: Input → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1) → Sigmoid
- FedProx local training: `SGD(lr=0.01, momentum=0.9, weight_decay=1e-5)`, proximal term μ=0.01
- Blockchain verification before model submission

**`main_tbfl_simulation.py`**: Main orchestration
- 100-round federated learning simulation across K=10 hospitals
- Weighted FedAvg aggregation (weighted by each client's local sample count)
- Performance metrics logging

**`sybil_attack_experiment.py`**: Security experiment (Sec. 5.4)
- Injects real Gaussian-noise Sybil nodes mid-training
- Compares an unprotected baseline against the blockchain-gated scenario across
  multiple seeds, with an independent-samples t-test on the resulting AUC

---

## 🪪 Identity Layer: DID, Verifiable Credentials & Zero-Knowledge Proofs

`FLRegistryZK.sol` implements the full Identity Layer described in the paper (Sec. 4.1),
on top of (not replacing) the simpler `FLRegistry.sol` used by the main FL simulation:

- **DIDs**: a participant's identity *is* its Ethereum address, resolvable off-chain as
  `did:ethr:<address>` (`src/identity.py::to_did`).
- **Verifiable Credentials**: the Trusted Issuer signs an EIP-712 typed credential
  `(subject, expiresAt)` **off-chain** (`src/identity.py::issue_credential`).
  `authorizeWorker` recovers the signer **on-chain** via `ecrecover` and only accepts a
  genuine Issuer signature; credentials carry a real expiration and can be revoked
  (`revokeCredential`).
- **Zero-Knowledge selective disclosure**: `submitUpdateZK` lets a credentialed
  participant prove Groth16 membership in a Merkle tree of credential commitments
  (`circuits/merkleMembership.circom`) **without revealing which institution it is**. A
  round-bound nullifier (a public output of the circuit) stops the same anonymous
  credential from submitting twice in one round.

**Scope note:** "anonymous" here means the ZK proof does not reveal *which* credentialed
institution is submitting. The Ethereum transaction sender is still a public field of the
transaction unless routed through a relayer/meta-transaction, which this PoC does not
implement. The Powers-of-Tau trusted setup generated by `scripts/zk/setup.sh` is a
single-contributor setup appropriate for local research/PoC use — **not** a real
production ceremony (see the warning in that script).

### Setup & demo

```bash
# One-time: compile the circuit, run the trusted setup, export the Solidity verifier
# (build artifacts are gitignored, so run this once after cloning)
npm run zk:setup

# Terminal A
npx hardhat node

# Terminal B
npx hardhat run scripts/deploy_zk.js --network localhost
python src/demo_did_vc_zk.py
```

`demo_did_vc_zk.py` walks through the paper's four-phase workflow end-to-end and prints
the result of each step: a hospital's EIP-712 credential being verified on-chain via
`ecrecover`, an unauthorized attacker being rejected, an anonymous ZK-proven submission
being accepted without revealing which of the credentialed hospitals sent it, a replayed
proof being blocked by the nullifier, and an attacker (whose secret was never added to the
registry) being unable to construct a valid proof at all.

---

## 🧪 Testing

There is currently no automated test suite (`npx hardhat test` will report no tests
found, since there is no `test/` directory yet — this is tracked as a contribution
opportunity below). Correctness is instead demonstrated by the standalone demo scripts,
each of which exercises both the accept and reject paths against a live local node:

- `python src/demo_security_mechanism.py` — `FLRegistry.sol` allowlist: authorized
  hospital accepted, unauthorized attacker rejected
- `python src/demo_did_vc_zk.py` — `FLRegistryZK.sol`: EIP-712 credential verified via
  `ecrecover`, anonymous ZK membership proof accepted, replay and non-member proofs rejected
- `python src/sybil_attack_experiment.py` — blockchain-gated vs. unprotected FedAvg under
  a real Sybil-node injection

---

## 📈 Performance Benchmarks

### Scalability Analysis

| Number of Clients | Round Time (s) | Gas per Client | Total Cost (100 rounds) |
|-------------------|----------------|----------------|-------------------------|
| 3 | 51.06 | 45,200 | $18.12 |
| 5 | 85.10 | 45,180 | $30.15 |
| 10 | 170.20 | 45,195 | $60.26 |

### Computational Overhead

| Operation | Time (seconds) | % of Total |
|-----------|----------------|------------|
| Local Training | 17.00 | 99.88% |
| Blockchain Verification | 0.02 | 0.12% |
| **Total** | **17.02** | **100%** |

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

**Contribution Areas**:
- 🐛 Bug fixes and testing (a `test/` suite for the contracts does not exist yet)
- 📄 The SQL cohort-selection query used to build `mortalidade_features.csv` from raw
  MIMIC-IV is not yet included in the repository
- 📚 Documentation improvements
- 🔬 New attack simulations
- 🚀 Performance optimizations
- 🏥 Additional clinical datasets

---

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tertulino2026,
      title={Trustworthy Blockchain-based Federated Learning for Electronic Health Records: Securing Participant Identity with Decentralized Identifiers and Verifiable Credentials}, 
      author={Rodrigo Tertulino and Ricardo Almeida and Laercio Alencar},
      year={2026},
      eprint={2602.02629},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.02629}, 
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MIMIC-IV Team** at MIT Laboratory for Computational Physiology
- **PhysioNet** for providing credentialed access to clinical data
- **Federal Institute of Education, Science, and Technology of Rio Grande do Norte (IFRN)** for computational resources
- **Software Engineering and Automation Research Laboratory** for institutional support
- **Hyperledger** and **Ethereum Foundation** for open-source blockchain tools

---

## 📧 Contact

**Rodrigo Tertulino**  
📧 Email: rodrigo.tertulino@ifrn.edu.br  
🔗 LinkedIn: [Rodrigo Tertulino](https://www.linkedin.com/in/rodrigo-tertulino-phd-06557962/).
🐙 GitHub: [@rodrigoronner](https://github.com/rodrigoronner)

**Research Group**: Software Engineering and Automation Research Laboratory  
🏛️ Institution: Federal Institute of Rio Grande do Norte (IFRN), Brazil

---

## ⚠️ Disclaimer

This software is provided for **research and educational purposes only**. It should not be used in production clinical environments without extensive additional validation, regulatory approval, and compliance verification. The authors assume no liability for any harm resulting from the use of this software.

**Data Privacy**: Users must comply with all applicable data protection regulations (e.g., GDPR, HIPAA) when working with electronic health records. The MIMIC-IV dataset is subject to the PhysioNet Data Use Agreement.

---

<div align="center">

**🌟 Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/rodrigoronner/TBFL-EHR-Framework?style=social)](https://github.com/rodrigoronner/TBFL-EHR-Framework)
[![GitHub forks](https://img.shields.io/github/forks/rodrigoronner/TBFL-EHR-Framework?style=social)](https://github.com/rodrigoronner/TBFL-EHR-Framework/fork)

</div>
