"""
Camada de coordenação (v2): implanta o FLRegistry em um nó Hardhat local
e registra gas e latência reais de cada transação.
"""
import hashlib
import json
import os
import time

from web3 import Web3

ABI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FLRegistry.json')


def commitment(flat_weights):
    """Compromisso de uma atualização: SHA-256 dos parâmetros float32, em hexadecimal."""
    return hashlib.sha256(flat_weights.numpy().tobytes()).hexdigest()


class Chain:
    def __init__(self, account_offset=0, rpc_url='http://127.0.0.1:8545'):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError("Nó Hardhat indisponível em " + rpc_url)
        art = json.load(open(ABI_PATH))
        accts = self.w3.eth.accounts
        # Cada processo usa um bloco próprio de contas para evitar conflito de nonce
        self.issuer = accts[account_offset]
        self.pool = accts[account_offset + 1:]
        factory = self.w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
        tx = factory.constructor().transact({'from': self.issuer})
        rcpt = self.w3.eth.wait_for_transaction_receipt(tx)
        self.contract = self.w3.eth.contract(address=rcpt.contractAddress, abi=art['abi'])
        self.deploy_gas = rcpt.gasUsed
        self.log = []  # um registro por tentativa de transação

    def _send(self, fn, sender, kind, round_idx):
        """Envia a transação; devolve o recibo se ela foi incluída sem revert, senão None."""
        t0 = time.perf_counter()
        try:
            tx = fn.transact({'from': sender})
            rcpt = self.w3.eth.wait_for_transaction_receipt(tx)
            ok, gas = rcpt.status == 1, rcpt.gasUsed
        except Exception:
            # Revert detectado na estimativa de gas: a transação não é minerada
            rcpt, ok, gas = None, False, 0
        dt = time.perf_counter() - t0
        self.log.append({'round': round_idx, 'kind': kind, 'sender': sender,
                         'accepted': ok, 'gas': gas, 'latency_s': dt})
        return rcpt if ok else None

    def authorize(self, worker, round_idx=0):
        """Issuer registra a posse de VC válida (verificação da VC é off-chain)."""
        return self._send(self.contract.functions.authorizeWorker(worker),
                          self.issuer, 'authorize', round_idx)

    def self_authorize(self, worker, round_idx=0):
        """Tentativa de um Sybil de se autorizar sem ser o Issuer."""
        return self._send(self.contract.functions.authorizeWorker(worker),
                          worker, 'sybil_authorize', round_idx)

    def submit(self, worker, flat_weights, round_idx, kind='submit'):
        """Lado do cliente: registra o compromisso da atualização. Devolve o recibo ou None."""
        return self._send(self.contract.functions.submitUpdate(commitment(flat_weights)),
                          worker, kind, round_idx)

    def verify(self, rcpt, sender, flat_weights):
        """
        Lado do agregador (G4): lê o compromisso admitido no evento ModelUpdated do recibo,
        confere o remetente da transação e compara com o SHA-256 da atualização recebida
        pelo canal off-chain. Atualizações sem compromisso correspondente são descartadas.
        """
        if rcpt is None or rcpt['from'] != sender:
            return False
        events = self.contract.events.ModelUpdated().process_receipt(rcpt)
        return len(events) == 1 and events[0]['args']['newHash'] == commitment(flat_weights)
