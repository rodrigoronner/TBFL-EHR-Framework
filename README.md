# TBFL-EHR: Trustworthy Blockchain-based Federated Learning for Electronic Health Records

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardhat](https://img.shields.io/badge/built%20with-Hardhat-FFDB1C.svg)](https://hardhat.org/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://arxiv.org/abs/2602.02629)

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
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Hospital A  │  │  Hospital B  │  │  Hospital C  │      │
│  │   (Client)   │  │   (Client)   │  │   (Client)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│              BLOCKCHAIN LAYER (Identity Verification)        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FLRegistry.sol (Ethereum Smart Contract)            │   │
│  │  • DID/VC Verification                               │   │
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
   - Manages participant authentication via `FLRegistry.sol` smart contract
   - Enforces access control through DID/VC verification
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
- `torch>=1.12.0`: Deep learning framework
- `web3>=5.31.0`: Ethereum blockchain interaction
- `pandas>=1.5.0`: Data manipulation
- `scikit-learn>=1.1.0`: Machine learning utilities
- `imbalanced-learn>=0.9.0`: SMOTETomek implementation
- `psycopg2-binary>=2.9.0`: PostgreSQL connector

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
2. Execute SQL cohort selection query (see SQL logic embedded in `src/main_tbfl_simulation.py`)
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

**Execution Flow** (100 rounds):

```
[Round 1/100] Blockchain Handshake... ✓
[Round 1/100] Loading MIMIC-IV data... ✓ (546,028 samples)
[Round 1/100] Applying SMOTETomek balancing... ✓
[Round 1/100] Hospital A - Local Training... ✓ (Loss: 0.312, Acc: 0.856)
[Round 1/100] Hospital A - Blockchain Verification... ✓ (Gas: 45,231)
[Round 1/100] Hospital B - Local Training... ✓ (Loss: 0.298, Acc: 0.862)
[Round 1/100] Hospital B - Blockchain Verification... ✓ (Gas: 45,187)
[Round 1/100] Hospital C - Local Training... ✓ (Loss: 0.305, Acc: 0.859)
[Round 1/100] Hospital C - Blockchain Verification... ✓ (Gas: 45,209)
[Round 1/100] Global Aggregation... ✓
[Round 1/100] Global Model - Loss: 0.305, Acc: 0.859, AUC: 0.891
...
[Round 100/100] Global Model - Loss: 0.272, Acc: 0.881, AUC: 0.954
Simulation Complete! Results logged in console output.
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
├── contracts/                 # Ethereum Smart Contracts
│   └── FLRegistry.sol        # Main access control contract
│
├── scripts/                   # Deployment and utilities
│   └── deploy.js             # Contract deployment script
│
├── src/                       # Federated Learning Core
│   ├── main_tbfl_simulation.py      # Main execution script
│   ├── blockchain_manager.py        # Web3 interface
│   └── cliente_fl.py                # FL client (hospital)
│
├── data/                      # Data directory
│   ├── mortalidade_features.csv.zip # Compressed preprocessed dataset
│   ├── mortalidade_features.csv     # Extracted dataset (after unzip)
│   └── raw/                         # Raw MIMIC-IV files (optional)
│
├── hardhat.config.js          # Hardhat configuration
├── package.json               # Node.js dependencies
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                  # This file
```

### Directory Descriptions

- **`contracts/`**: Contains Solidity smart contracts for blockchain-based access control
  - `FLRegistry.sol`: Implements DID/VC verification and authorization registry

- **`scripts/`**: Deployment automation and utility scripts
  - `deploy.js`: Deploys FLRegistry contract to local or testnet Ethereum network

- **`src/`**: Core federated learning implementation
  - `main_tbfl_simulation.py`: Orchestrates the complete TBFL workflow (100 rounds)
  - `blockchain_manager.py`: Web3.py interface for smart contract interaction
  - `cliente_fl.py`: Hospital client implementing MLP training and FedProx optimization

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
NUM_CLIENTS=3
NUM_ROUNDS=100
LEARNING_RATE=0.001
FEDPROX_MU=0.01
```

### Hardhat Network Configuration

Edit `hardhat.config.js` for custom networks:

```javascript
module.exports = {
  solidity: "0.8.19",
  networks: {
    hardhat: {
      chainId: 1337,
      mining: {
        auto: true,
        interval: 0  // Instant mining for testing
      }
    },
    localhost: {
      url: "http://127.0.0.1:8545"
    },
    // Testnet deployment (optional)
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
// Authorize hospital to participate
function authorizeWorker(address worker) external onlyIssuer

// Submit model hash after local training
function submitUpdate(bytes32 modelHash) external onlyAuthorized

// Query authorization status
function isAuthorized(address worker) external view returns (bool)
```

**Events**:
```solidity
event WorkerAuthorized(address indexed worker, uint256 timestamp);
event UpdateSubmitted(address indexed worker, bytes32 modelHash, uint256 round);
```

### Python Modules

**`blockchain_manager.py`**: Web3 interface
- ABI loading and contract interaction
- Transaction signing with private keys
- Gas estimation and monitoring
- Connection to local Hardhat node

**`cliente_fl.py`**: Federated Learning client
- MLP architecture (3 hidden layers: 128→64→32 neurons)
- FedProx optimizer with proximal term μ=0.01
- Local SMOTETomek balancing for class imbalance
- Blockchain verification before model submission

**`main_tbfl_simulation.py`**: Main orchestration
- 100-round federated learning simulation
- Multi-client coordination (3 hospitals by default)
- Performance metrics logging
- Security validation (Sybil attack prevention)

---

## 🧪 Testing

### Smart Contract Tests

```bash
# Run all contract tests
npx hardhat test

# Run with gas reporting
REPORT_GAS=true npx hardhat test
```

**Test Coverage**:
- ✅ Authorization workflow
- ✅ Unauthorized access prevention
- ✅ Model submission validation
- ✅ Event emission verification

**Note**: Additional test files for Python components will be added in future releases.

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
- 🐛 Bug fixes and testing
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
