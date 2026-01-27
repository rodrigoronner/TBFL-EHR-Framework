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
