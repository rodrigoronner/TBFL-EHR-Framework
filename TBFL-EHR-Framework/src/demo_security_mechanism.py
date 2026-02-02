import torch
import pandas as pd
import json
import os
from web3 import Web3

"""
demo_security_mechanism.py

This script provides a standalone Proof of Concept (PoC) demonstrating the 
Identity-First security mechanism proposed in the TBFL framework.

It simulates:
1. A legitimate node (Hospital A) receiving a Verifiable Credential.
2. An illegitimate node (Attacker) attempting to submit updates without a credential.
3. The Smart Contract enforcing access control.
"""

# ================= CONFIGURATION =================
RPC_URL = 'http://127.0.0.1:8545'
# NOTE: Update this address after running 'npx hardhat run scripts/deploy.js'
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3' 
# =================================================

def load_contract_abi():
    """Helper to load the ABI from the Hardhat artifacts folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_path = os.path.join(script_dir, 'artifacts', 'contracts', 'FLRegistry.sol', 'FLRegistry.json')
    
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"❌ ABI Artifact not found at {artifact_path}. Did you compile the contracts?")
        
    with open(artifact_path) as f:
        return json.load(f)['abi']

def main():
    # 1. Connection to Local Blockchain (Hardhat Node)
    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        if not w3.is_connected():
            raise Exception("Node not connected")
    except Exception as e:
        print(f"❌ Failed to connect to Blockchain: {e}")
        return

    # Load Contract
    try:
        abi = load_contract_abi()
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
        print(f"🔗 Connected to Smart Contract at {CONTRACT_ADDRESS}")
    except Exception as e:
        print(f"❌ Contract Error: {e}")
        return

    # 2. Identity Simulation (DIDs)
    # Hardhat provides pre-funded accounts. We assign roles:
    issuer = w3.eth.accounts[0]     # Trusted Issuer (e.g., Ministry of Health)
    hospital_A = w3.eth.accounts[1] # Legitimate Participant
    hospital_B = w3.eth.accounts[2] # Legitimate Participant
    attacker = w3.eth.accounts[3]   # Malicious Node (No VC issued)

    print("\n--- 🏛️  PHASE 1: Credential Issuance (Onboarding) ---")
    
    # The Issuer authorizes legitimate hospitals on-chain (Simulating VC issuance)
    print(f"🔹 Issuer authorizing Hospital A ({hospital_A[:6]}...)...")
    tx_hash = contract.functions.authorizeWorker(hospital_A).transact({'from': issuer})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("   ✅ Authorization Confirmed.")

    print(f"🔹 Issuer authorizing Hospital B ({hospital_B[:6]}...)...")
    tx_hash = contract.functions.authorizeWorker(hospital_B).transact({'from': issuer})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("   ✅ Authorization Confirmed.")
    
    print(f"⚠️  Attacker ({attacker[:6]}...) is NOT authorized.")

    # 3. Data Loading (Placeholder)
    # In a real scenario, this would load the MIMIC-IV dataset
    # data = pd.read_csv('./data/mimic_iv_processed.csv')
    print("\n--- 🏥 PHASE 2: Local Training & Submission ---")

    # 4. Simulation Function
    def simulate_training_and_submission(worker_account, label):
        print(f"\n🔄 [{label}] Starting local training...")
        
        # ... Standard PyTorch training logic would go here ...
        # ... net.train() ...
        
        # Simulate the generation of a Model Update Hash (IPFS CID)
        # In practice, this hash points to the weights stored off-chain
        model_weights_hash = f"QmHashIPFS_Simulated_{worker_account[:5]}_{label}"
        
        print(f"   Generated Model Hash: {model_weights_hash}")
        print(f"   Attempting to submit hash to Blockchain...")
        
        # 5. Blockchain Submission Attempt
        try:
            # The Smart Contract checks: require(authorized[msg.sender], "Access Denied");
            tx = contract.functions.submitUpdate(model_weights_hash).transact({'from': worker_account})
            receipt = w3.eth.wait_for_transaction_receipt(tx)
            
            print(f"   ✅ SUCCESS: Update accepted in block {receipt['blockNumber']}.")
            print(f"   Gas Used: {receipt['gasUsed']}")
            
        except Exception as e:
            # If the node is not authorized, the transaction reverts
            print(f"   ⛔ BLOCKED: Transaction reverted by Smart Contract.")
            print(f"   Reason: Access Denied / Caller is not authorized.")

    # EXECUTE SCENARIOS
    
    # Scenario A: Legitimate Hospital
    simulate_training_and_submission(hospital_A, "Hospital A (Legit)")
    
    # Scenario B: Sybil Attacker
    simulate_training_and_submission(attacker, "Sybil Attacker")

if __name__ == "__main__":
    main()
