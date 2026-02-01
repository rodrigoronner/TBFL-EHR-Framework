import json
from web3 import Web3
import os

class BlockchainManager:
    def __init__(self, contract_address, rpc_url='http://127.0.0.1:8545'):
        # 1. Conexão com Hardhat Local
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise Exception("❌ Falha ao conectar com Blockchain. Verifique se 'npx hardhat node' está rodando em outro terminal.")

        # 2. Correção Robusta do Caminho para o ABI
        # Pega o diretório onde ESTE arquivo (blockchain_manager.py) está salvo (ex: .../TBFL_Project/scripts)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Sobe um nível para chegar na raiz do projeto (ex: .../TBFL_Project)
        project_root = os.path.dirname(script_dir)
        
        # Monta o caminho para artifacts/contracts/...
        abi_path = os.path.join(project_root, 'artifacts', 'contracts', 'FLRegistry.sol', 'FLRegistry.json')
        
        # Debug: Mostra onde o script está procurando (caso dê erro de novo)
        # print(f"🔍 Procurando ABI em: {abi_path}") 
        
        if not os.path.exists(abi_path):
            raise Exception(f"❌ Arquivo ABI não encontrado no caminho:\n{abi_path}\nCertifique-se de que você rodou 'npx hardhat compile' na raiz do projeto.")
        
        try:
            with open(abi_path) as f:
                contract_json = json.load(f)
                abi = contract_json['abi']
        except Exception as e:
            raise Exception(f"❌ Erro ao ler o arquivo JSON do ABI: {e}")
        
        self.contract = self.w3.eth.contract(address=contract_address, abi=abi)
        
        # 3. Configurar Contas
        # Accounts[0] = Ministério da Saúde (Issuer)
        self.issuer = self.w3.eth.accounts[0]
        self.accounts = self.w3.eth.accounts
        
        print(f"🔗 Blockchain Conectado. Contrato: {contract_address}")

    def get_account(self, index):
        return self.accounts[index]

    def issue_credential(self, worker_address):
        """Simula a emissão de uma VC (Verifiable Credential) on-chain."""
        print(f"🏛️  Issuer emitindo credencial para: {worker_address[:8]}...")
        try:
            tx = self.contract.functions.authorizeWorker(worker_address).transact({'from': self.issuer})
            self.w3.eth.wait_for_transaction_receipt(tx)
            print(f"   ✅ Credencial registrada com sucesso.")
        except Exception as e:
            print(f"   ❌ Erro ao emitir credencial: {e}")

    def submit_hash(self, worker_address, model_hash):
        """
        Tenta enviar o hash do modelo. 
        Se o worker não tiver credencial, o Smart Contract REVERTE a transação.
        """
        try:
            tx = self.contract.functions.submitUpdate(model_hash).transact({'from': worker_address})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx)
            gas_used = receipt['gasUsed']
            return True, gas_used
        except Exception as e:
            # Aqui capturamos o erro "Access Denied" do Solidity ou erro de conexão
            # print(f"Debug Erro: {e}") 
            return False, 0
