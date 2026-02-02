import json
from web3 import Web3
import os

class BlockchainManager:
    """
    Manages interactions with the Ethereum-based blockchain network (Hardhat).
    Responsible for connecting to the node, loading smart contracts, 
    managing identity (DID/VC issuance), and handling transaction submissions.
    """

    def __init__(self, contract_address, rpc_url='http://127.0.0.1:8545'):
        # 1. Initialize Web3 connection to the local Hardhat RPC
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise Exception("❌ Failed to connect to Blockchain. Ensure 'npx hardhat node' is running in a separate terminal.")

        # 2. Dynamic Path Resolution for Contract ABI
        # This ensures the script works regardless of where it is executed from within the project structure.
        
        # Get the directory where THIS script (blockchain_manager.py) is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Move up one level to reach the project root (e.g., .../TBFL_Project)
        project_root = os.path.dirname(script_dir)
        
        # Construct the absolute path to the compiled ABI artifact (standard Hardhat path)
        abi_path = os.path.join(project_root, 'artifacts', 'contracts', 'FLRegistry.sol', 'FLRegistry.json')
        
        # Debug: Print path if file is not found (useful for troubleshooting during review)
        # print(f"🔍 Searching for ABI at: {abi_path}") 
        
        if not os.path.exists(abi_path):
            raise Exception(f"❌ ABI file not found at:\n{abi_path}\nPlease run 'npx hardhat compile' at the project root to generate artifacts.")
        
        try:
            with open(abi_path) as f:
                contract_json = json.load(f)
                abi = contract_json['abi']
        except Exception as e:
            raise Exception(f"❌ Error loading ABI JSON file: {e}")
        
        # Initialize the Contract Instance
        self.contract = self.w3.eth.contract(address=contract_address, abi=abi)
        
        # 3. Configure Roles and Accounts
        # Accounts[0] is designated as the 'Trusted Issuer' (e.g., Ministry of Health)
        # capable of authorizing other nodes.
        self.issuer = self.w3.eth.accounts[0]
        self.accounts = self.w3.eth.accounts
        
        print(f"🔗 Blockchain Connected. Contract loaded at: {contract_address}")

    def get_account(self, index):
        """Retrieve a specific blockchain account by index."""
        return self.accounts[index]

    def issue_credential(self, worker_address):
        """
        Simulates the issuance of a Verifiable Credential (VC) on-chain.
        The Trusted Issuer authorizes a worker address to participate in the federation.
        """
        print(f"🏛️  Trusted Issuer issuing credential to: {worker_address[:8]}...")
        try:
            # Calls the 'authorizeWorker' function in the Smart Contract
            tx = self.contract.functions.authorizeWorker(worker_address).transact({'from': self.issuer})
            self.w3.eth.wait_for_transaction_receipt(tx)
            print(f"   ✅ Credential successfully registered on the ledger.")
        except Exception as e:
            print(f"   ❌ Error issuing credential: {e}")

    def submit_hash(self, worker_address, model_hash):
        """
        Attempts to submit a model update hash (IPFS CID) to the blockchain.
        
        Security Enforcement:
        If the worker address does not hold a valid credential (VC), the Smart Contract 
        will REVERT the transaction, effectively blocking the update.
        
        Returns:
            success (bool): True if transaction succeeded.
            gas_used (int): Amount of gas consumed (for economic analysis).
        """
        try:
            # Calls the 'submitUpdate' function in the Smart Contract
            tx = self.contract.functions.submitUpdate(model_hash).transact({'from': worker_address})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx)
            
            # Capture gas usage to calculate operational costs reported in the paper
            gas_used = receipt['gasUsed']
            return True, gas_used
        except Exception as e:
            # Catches Solidity 'Access Denied' errors or connection issues
            # print(f"Debug Error: {e}") 
            return False, 0
