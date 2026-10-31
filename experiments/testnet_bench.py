"""
Benchmark dos registros em uma testnet pública (Sepolia ou Arbitrum Sepolia).

Mede, em uma rede peer-to-peer real, o que o nó Hardhat local não mede: o tempo até a
inclusão em bloco, o preço efetivo do gas e o tempo de uma rodada em que todos os clientes
submetem compromissos ao mesmo tempo. Repete os casos de ataque da Proposição 1.

A chave da conta financiadora vem de TBFL_TESTNET_KEY ou do arquivo ~/tbfl_work/testnet_key
e nunca é gravada no repositório. As contas dos clientes e do atacante são derivadas dela de
forma determinística e financiadas com uma pequena quantia no início.

  python testnet_bench.py --network sepolia --workers 5 --rounds 10
  python testnet_bench.py --network arbsepolia --workers 5 --rounds 10
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from eth_account import Account
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORKS = {
    'sepolia': 'https://ethereum-sepolia-rpc.publicnode.com',
    'arbsepolia': 'https://sepolia-rollup.arbitrum.io/rpc',
}
EXPLORER = {
    'sepolia': 'https://sepolia.etherscan.io',
    'arbsepolia': 'https://sepolia.arbiscan.io',
}


def load_key():
    key = os.environ.get('TBFL_TESTNET_KEY')
    if not key:
        key = open(os.path.expanduser('~/tbfl_work/testnet_key')).read().strip()
    return Account.from_key(key)


def derived(master, label):
    return Account.from_key(Web3.keccak(master.key + label.encode()))


class Net:
    def __init__(self, url):
        self.w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 60}))
        self.chain_id = self.w3.eth.chain_id
        self.nonces = {}
        self.log = []

    def fees(self):
        base = self.w3.eth.get_block('latest')['baseFeePerGas']
        tip = max(self.w3.eth.max_priority_fee, 1)
        return {'maxFeePerGas': 2 * base + tip, 'maxPriorityFeePerGas': tip}

    def nonce(self, acct):
        if acct.address not in self.nonces:
            self.nonces[acct.address] = self.w3.eth.get_transaction_count(acct.address, 'pending')
        n = self.nonces[acct.address]
        self.nonces[acct.address] += 1
        return n

    def send(self, acct, tx):
        """Assina e envia; devolve (hash, instante do envio) ou (None, motivo) se reverte na estimativa."""
        tx = {k: v for k, v in tx.items()
              if k not in ('gas', 'nonce', 'gasPrice', 'maxFeePerGas', 'maxPriorityFeePerGas')}
        tx.update(chainId=self.chain_id, **{'from': acct.address}, **self.fees())
        try:
            tx['gas'] = int(self.w3.eth.estimate_gas(tx) * 1.2)
        except Exception as e:
            return None, str(e)[:80]
        tx['nonce'] = self.nonce(acct)
        signed = acct.sign_transaction(tx)
        t0 = time.time()
        return self.w3.eth.send_raw_transaction(signed.raw_transaction), t0

    def wait(self, h, t0, op, contract, round_idx=None):
        rcpt = self.w3.eth.wait_for_transaction_receipt(h, timeout=600, poll_latency=0.25)
        dt = time.time() - t0
        self.log.append({'contract': contract, 'op': op, 'round': round_idx,
                         'accepted': rcpt.status == 1, 'gas': rcpt.gasUsed,
                         'gas_price_gwei': rcpt.effectiveGasPrice / 1e9,
                         'block': rcpt.blockNumber, 'latency_s': dt, 'tx': h.hex()})
        return rcpt

    def run(self, acct, fn_tx, op, contract, round_idx=None):
        h, t0 = self.send(acct, fn_tx)
        if h is None:
            self.log.append({'contract': contract, 'op': op, 'round': round_idx,
                             'accepted': False, 'gas': 0, 'gas_price_gwei': np.nan,
                             'block': np.nan, 'latency_s': np.nan, 'tx': '', 'reason': t0})
            return None
        return self.wait(h, t0, op, contract, round_idx)

    def deploy(self, acct, name, args=()):
        art = json.load(open(os.path.join(HERE, f'{name}.json')))
        f = self.w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
        rcpt = self.run(acct, f.constructor(*args).build_transaction({'from': acct.address}),
                        'deploy', name)
        return self.w3.eth.contract(address=rcpt.contractAddress, abi=art['abi'])


def credential(net, contract, issuer, subject, expires_at, issued_at):
    typed = {
        'types': {
            'EIP712Domain': [{'name': 'name', 'type': 'string'},
                             {'name': 'version', 'type': 'string'},
                             {'name': 'chainId', 'type': 'uint256'},
                             {'name': 'verifyingContract', 'type': 'address'}],
            'HospitalCredential': [{'name': 'subject', 'type': 'address'},
                                   {'name': 'expiresAt', 'type': 'uint256'},
                                   {'name': 'issuedAt', 'type': 'uint256'}]},
        'primaryType': 'HospitalCredential',
        'domain': {'name': 'TBFL-FLRegistryZK', 'version': '1', 'chainId': net.chain_id,
                   'verifyingContract': contract.address},
        'message': {'subject': subject, 'expiresAt': expires_at, 'issuedAt': issued_at}}
    return Account.sign_typed_data(issuer.key, full_message=typed).signature


def tx(fn, acct):
    return fn.build_transaction({'from': acct.address, 'gas': 0, 'nonce': 0})


def rounds(net, contract, name, workers, n_rounds, rng):
    """Cada rodada: todos os clientes enviam o compromisso ao mesmo tempo; mede até o último recibo."""
    per_round = []
    for r in range(1, n_rounds + 1):
        sent = []
        for wk in workers:
            upd = rng.standard_normal(4225).astype(np.float32).tobytes()
            h, t0 = net.send(wk, tx(contract.functions.submitUpdate(
                hashlib.sha256(upd).hexdigest()), wk))
            sent.append((h, t0, wk, upd))
        t_start = min(t0 for _, t0, _, _ in sent)
        ok_g4 = 0
        for h, t0, wk, upd in sent:
            rcpt = net.wait(h, t0, 'submit', name, r)
            ev = contract.events.ModelUpdated().process_receipt(rcpt)
            ok_g4 += int(rcpt['from'] == wk.address and
                         ev[0]['args']['newHash'] == hashlib.sha256(upd).hexdigest())
        per_round.append({'contract': name, 'round': r, 'round_time_s': time.time() - t_start,
                          'g4_verified': ok_g4, 'workers': len(workers)})
        print(f'  {name} rodada {r}: {per_round[-1]["round_time_s"]:.1f} s', flush=True)
    return per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', choices=list(NETWORKS), required=True)
    ap.add_argument('--rpc', default=None)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--fund_eth', type=float, default=0.002)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results'))
    a = ap.parse_args()

    net = Net(a.rpc or NETWORKS[a.network])
    issuer = load_key()
    workers = [derived(issuer, f'worker{i}') for i in range(a.workers)]
    attacker = derived(issuer, 'attacker')
    bal = net.w3.eth.get_balance(issuer.address) / 1e18
    print(f'{a.network} (chainId {net.chain_id}); emissor {issuer.address}: {bal:.4f} ETH', flush=True)
    if bal < a.fund_eth * (a.workers + 1) + 0.005:
        raise SystemExit('saldo insuficiente: financie o endereço do emissor por um faucet')

    # Financia clientes e atacante (o atacante precisa de saldo para que a recusa seja um revert)
    for acct in workers + [attacker]:
        need = Web3.to_wei(a.fund_eth, 'ether') - net.w3.eth.get_balance(acct.address)
        if need > 0:
            net.run(issuer, {'to': acct.address, 'value': need}, 'fund', '-')

    rng = np.random.default_rng(0)
    allow = net.deploy(issuer, 'FLRegistry')
    zk = net.deploy(issuer, 'FLRegistryZK', (issuer.address,))

    # Allowlist: o emissor autoriza; o atacante tenta se autorizar e submeter
    for wk in workers:
        net.run(issuer, tx(allow.functions.authorizeWorker(wk.address), issuer), 'authorize',
                'FLRegistry')
    net.run(attacker, tx(allow.functions.authorizeWorker(attacker.address), attacker),
            'sybil_self_authorize', 'FLRegistry')
    net.run(attacker, tx(allow.functions.submitUpdate('00' * 32), attacker), 'sybil_submit',
            'FLRegistry')

    # Credencial assinada: cada cliente retransmite a própria VC; casos de ataque de G2
    now = net.w3.eth.get_block('latest')['timestamp']
    exp = now + 365 * 24 * 3600
    vcs = {}
    for wk in workers:
        vcs[wk.address] = credential(net, zk, issuer, wk.address, exp, now)
        net.run(wk, tx(zk.functions.authorizeWorker(wk.address, exp, now, vcs[wk.address]), wk),
                'authorize', 'FLRegistryZK')
    forged = credential(net, zk, attacker, attacker.address, exp, now)
    net.run(attacker, tx(zk.functions.authorizeWorker(attacker.address, exp, now, forged),
                         attacker), 'forged_vc', 'FLRegistryZK')
    w0 = workers[0].address
    net.run(attacker, tx(zk.functions.authorizeWorker(attacker.address, exp, now, vcs[w0]),
                         attacker), 'replayed_vc_other_subject', 'FLRegistryZK')
    expired = credential(net, zk, issuer, attacker.address, now - 1, now - 2)
    net.run(attacker, tx(zk.functions.authorizeWorker(attacker.address, now - 1, now - 2, expired),
                         attacker), 'expired_vc', 'FLRegistryZK')
    net.run(attacker, tx(zk.functions.authorizeWorker(w0, exp, now, vcs[w0]), attacker),
            'replayed_vc_same_subject', 'FLRegistryZK')
    net.run(attacker, tx(zk.functions.submitUpdate('00' * 32), attacker), 'sybil_submit',
            'FLRegistryZK')

    per_round = rounds(net, allow, 'FLRegistry', workers, a.rounds, rng)
    per_round += rounds(net, zk, 'FLRegistryZK', workers, a.rounds, rng)

    wl = workers[-1].address
    rcpt = net.run(issuer, tx(zk.functions.revokeCredential(wl), issuer), 'revoke', 'FLRegistryZK')
    t_rev = net.w3.eth.get_block(rcpt.blockNumber)['timestamp']
    net.run(workers[-1], tx(zk.functions.submitUpdate('00' * 32), workers[-1]),
            'submit_after_revocation', 'FLRegistryZK')
    # Reenviar a credencial antiga (pública na cadeia) não pode desfazer a revogação
    net.run(attacker, tx(zk.functions.authorizeWorker(wl, exp, now, vcs[wl]), attacker),
            'replayed_vc_after_revocation', 'FLRegistryZK')
    net.run(workers[-1], tx(zk.functions.submitUpdate('00' * 32), workers[-1]),
            'submit_after_replay', 'FLRegistryZK')
    # Só uma credencial emitida depois da revogação reabilita o sujeito; espera o relógio da
    # cadeia passar do instante da revogação, porque a estimativa usa o último bloco
    while net.w3.eth.get_block('latest')['timestamp'] <= t_rev + 1:
        if net.chain_id == 31337:          # nó Hardhat local só mina quando há transação
            net.w3.provider.make_request('evm_mine', [])
        time.sleep(1)
    renewed = credential(net, zk, issuer, wl, exp, t_rev + 1)
    net.run(workers[-1], tx(zk.functions.authorizeWorker(wl, exp, t_rev + 1, renewed),
                            workers[-1]), 'reissued_vc_after_revocation', 'FLRegistryZK')
    net.run(workers[-1], tx(zk.functions.submitUpdate('00' * 32), workers[-1]),
            'submit_after_reissue', 'FLRegistryZK')

    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(net.log)
    df.to_csv(os.path.join(a.out, f'testnet_{a.network}.csv'), index=False)
    pd.DataFrame(per_round).to_csv(os.path.join(a.out, f'testnet_{a.network}_rounds.csv'),
                                   index=False)
    meta = {'network': a.network, 'chain_id': net.chain_id, 'explorer': EXPLORER[a.network],
            'issuer': issuer.address, 'workers': [w.address for w in workers],
            'attacker': attacker.address, 'FLRegistry': allow.address,
            'FLRegistryZK': zk.address, 'date': time.strftime('%Y-%m-%d %H:%M %Z')}
    json.dump(meta, open(os.path.join(a.out, f'testnet_{a.network}_meta.json'), 'w'), indent=1)
    summ = df.groupby(['contract', 'op']).agg(n=('gas', 'size'), accepted=('accepted', 'sum'),
                                              gas_median=('gas', 'median'),
                                              latency_s=('latency_s', 'median'),
                                              gwei=('gas_price_gwei', 'median'))
    print(summ.to_string())
    print(pd.DataFrame(per_round).groupby('contract').round_time_s.describe().to_string())


if __name__ == '__main__':
    main()
