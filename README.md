# TBFL-EHR: Trustworthy Blockchain-based Federated Learning for Electronic Health Records

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardhat](https://img.shields.io/badge/built%20with-Hardhat-FFDB1C.svg)](https://hardhat.org/)

This repository contains the official implementation of the paper **"Trustworthy Blockchain-based Federated Learning for Electronic Health Records: Securing Participant Identity with Decentralized Identifiers and Verifiable Credentials"**, submitted to *ACM Distributed Ledger Technologies: Research and Practice*.

## 📄 Abstract

We propose a secure Federated Learning architecture that integrates Ethereum-based Smart Contracts to enforce strict identity governance (DIDs/VCs) before model aggregation. Validated on the **MIMIC-IV** dataset, the framework mitigates 100% of Sybil attacks with negligible computational overhead (<0.2%), ensuring robust mortality prediction (AUC 0.954) in adversarial healthcare environments.

## 🏗 Architecture

The system consists of three main layers:
1.  **Blockchain Layer (Hardhat):** Manages access control via the `AccessControl.sol` smart contract.
2.  **Federated Learning Layer (PyTorch):** Implements the FedProx algorithm with local `SMOTETomek` balancing.
3.  **Data Pipeline:** Processes MIMIC-IV clinical data (Note: Data is not included due to privacy restrictions).

## 🚀 Installation

### Prerequisites
* Python 3.8+
* Node.js & NPM
* Access to MIMIC-IV Database (Credentialed Access required via PhysioNet)

### 1. Clone the repository
```bash
git clone [https://github.com/SEU_USUARIO/TBFL-EHR-Framework.git](https://github.com/SEU_USUARIO/TBFL-EHR-Framework.git)
cd TBFL-EHR-Framework

## 🚀 Installation and Configuration
1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/TBFL-EHR-Framework.git](https://github.com/YOUR_USERNAME/TBFL-EHR-Framework.git)
cd TBFL-EHR-Framework

2. Configure Blockchain Environment (Hardhat)
Install the necessary Node.js packages defined in package.json:

Bash
npm install

3. Configure Python Environment (Federated Learning)
Create and activate a virtual environment, then install the dependencies:

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## 🧪 How to Replicate the Experiment (Step-by-Step)
Step 1: Data Acquisition (MIMIC-IV)
Due to the PhysioNet Data Use Agreement (DUA), raw patient data cannot be shared in this repository.

Obtain credentialed access to MIMIC-IV v2.2 at PhysioNet.

Perform cohort selection and feature extraction as described in the paper methodology.

Save the resulting dataset as mortalidade_features.csv.

Move the file into the data/ directory.

Step 2: Start Local Blockchain Node
Open a terminal window (Terminal A) and start the local Hardhat Ethereum node. Keep this terminal running.

Bash
npx hardhat node

Output: Started HTTP and WebSocket JSON-RPC server at https://www.google.com/search?q=http://127.0.0.1:8545/

Step 3: Deploy Smart Contract
Open a second terminal (Terminal B), ensure you are in the project root, and deploy the contract:

Bash
npx hardhat run scripts/deploy.js --network localhost

⚠️ CRITICAL: The terminal will output the deployed contract address (e.g., 0x5FbDB...).

Copy this address.

Open src/main_tbfl_simulation.py.

Update the CONTRACT_ADDRESS variable with the new address:

Python
# src/main_tbfl_simulation.py
CONTRACT_ADDRESS = '0x5FbDB...' # Paste your address here
Step 4: Execute Simulation
In Terminal B (with Python venv activated), run the main simulation script:

Bash
python src/main_tbfl_simulation.py
Expected Output
The script will perform the following actions over 100 communication rounds:

Blockchain Handshake: Connects to the local node and FLRegistry.

Data Loading: Loads and balances MIMIC-IV data (SMOTETomek).

Federated Loop:

Clients train local models (Off-chain).

Clients submit model hashes to the Blockchain (On-chain verification).

Server aggregates validated models.

Results: A CSV file containing metrics (Loss, Accuracy, AUC, Gas Used, Latency) will be generated in the root directory.

🧩 Component Details
FLRegistry.sol: Manages the allowlist of authorized hospitals and logs model updates.

blockchain_manager.py: Handles the ABI resolution and transaction signing using web3.py.

cliente_fl.py: Implements the MLP architecture and the FedProx optimizer to handle Non-IID data.




