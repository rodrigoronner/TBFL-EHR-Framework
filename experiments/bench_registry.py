"""
Microbenchmark das variantes do registro on-chain (v2).

  FLRegistry        allowlist mínima usada na grade experimental (run_experiment.py)
  FLRegistryEvents  allowlist com eventos indexados (versão publicada no repositório)
  FLRegistryZK      credencial EIP-712 assinada pelo Issuer, verificada on-chain via ecrecover,
                    com expiração e revogação (caminho nomeado; o caminho Groth16 não é medido)

Para cada variante: gas de implantação, autorização, submissão e revogação, e o resultado
de tentativas adversariais. Requer um nó Hardhat em 127.0.0.1:8545 sem outros processos
usando as mesmas contas.

  python bench_registry.py --workers 10 --rounds 20
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))


class Bench:
    def __init__(self, w3, name, issuer, ctor_args=()):
        art = json.load(open(os.path.join(HERE, f'{name}.json')))
        self.w3, self.name, self.issuer = w3, name, issuer
        f = w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
        rcpt = w3.eth.wait_for_transaction_receipt(
            f.constructor(*ctor_args).transact({'from': issuer}))
        self.c = w3.eth.contract(address=rcpt.contractAddress, abi=art['abi'])
        self.log = [{'contract': name, 'op': 'deploy', 'accepted': True,
                     'gas': rcpt.gasUsed, 'latency_s': np.nan}]

    def send(self, op, fn, sender):
        t0 = time.perf_counter()
        try:
            rcpt = self.w3.eth.wait_for_transaction_receipt(fn.transact({'from': sender}))
            ok, gas = rcpt.status == 1, rcpt.gasUsed
        except Exception:
            ok, gas = False, 0  # revert na estimativa de gas: não minerada
        self.log.append({'contract': self.name, 'op': op, 'accepted': ok, 'gas': gas,
                         'latency_s': time.perf_counter() - t0})
        return ok


def sign_vc(w3, contract, issuer, subject, expires_at):
    """Credencial EIP-712 HospitalCredential(subject, expiresAt), assinada pela conta `issuer`."""
    typed = {
        'types': {
            'EIP712Domain': [{'name': 'name', 'type': 'string'},
                             {'name': 'version', 'type': 'string'},
                             {'name': 'chainId', 'type': 'uint256'},
                             {'name': 'verifyingContract', 'type': 'address'}],
            'HospitalCredential': [{'name': 'subject', 'type': 'address'},
                                   {'name': 'expiresAt', 'type': 'uint256'}]},
        'primaryType': 'HospitalCredential',
        'domain': {'name': 'TBFL-FLRegistryZK', 'version': '1',
                   'chainId': w3.eth.chain_id, 'verifyingContract': contract.address},
        'message': {'subject': subject, 'expiresAt': expires_at}}
    r = w3.provider.make_request('eth_signTypedData_v4', [issuer, json.dumps(typed)])
    return bytes.fromhex(r['result'][2:])


def digest(rng):
    return hashlib.sha256(rng.standard_normal(4225).astype(np.float32).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'bench_registry.csv'))
    a = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    acc = w3.eth.accounts[a.offset:]
    issuer, workers, attacker = acc[0], acc[1:a.workers + 1], acc[a.workers + 1]
    rng = np.random.default_rng(0)
    logs = []

    # Allowlists: o Issuer autoriza cada endereço; o atacante tenta se autorizar
    for name in ['FLRegistry', 'FLRegistryEvents']:
        b = Bench(w3, name, issuer)
        for wk in workers:
            b.send('authorize', b.c.functions.authorizeWorker(wk), issuer)
        b.send('sybil_self_authorize', b.c.functions.authorizeWorker(attacker), attacker)
        for _ in range(a.rounds):
            for wk in workers:
                b.send('submit', b.c.functions.submitUpdate(digest(rng)), wk)
            b.send('sybil_submit', b.c.functions.submitUpdate(digest(rng)), attacker)
        logs += b.log

    # Credencial assinada: qualquer um retransmite a VC; o contrato verifica a assinatura
    b = Bench(w3, 'FLRegistryZK', issuer, ctor_args=(issuer,))
    now = w3.eth.get_block('latest').timestamp
    exp = now + 365 * 24 * 3600
    vcs = {wk: sign_vc(w3, b.c, issuer, wk, exp) for wk in workers}
    for wk in workers:
        b.send('authorize', b.c.functions.authorizeWorker(wk, exp, vcs[wk]), wk)
    forged = sign_vc(w3, b.c, attacker, attacker, exp)
    b.send('forged_vc', b.c.functions.authorizeWorker(attacker, exp, forged), attacker)
    b.send('replayed_vc_other_subject',
           b.c.functions.authorizeWorker(attacker, exp, vcs[workers[0]]), attacker)
    expired = sign_vc(w3, b.c, issuer, attacker, now - 1)
    b.send('expired_vc', b.c.functions.authorizeWorker(attacker, now - 1, expired), attacker)
    for _ in range(a.rounds):
        for wk in workers:
            b.send('submit', b.c.functions.submitUpdate(digest(rng)), wk)
        b.send('sybil_submit', b.c.functions.submitUpdate(digest(rng)), attacker)
    b.send('revoke', b.c.functions.revokeCredential(workers[-1]), issuer)
    b.send('submit_after_revocation', b.c.functions.submitUpdate(digest(rng)), workers[-1])
    logs += b.log

    df = pd.DataFrame(logs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = df.groupby(['contract', 'op']).agg(n=('gas', 'size'), accepted=('accepted', 'sum'),
                                              gas_mean=('gas', 'mean'), gas_max=('gas', 'max'),
                                              latency_ms=('latency_s', lambda s: 1e3 * s.mean()))
    summ.round(1).to_csv(a.out.replace('.csv', '_summary.csv'))
    print(summ.round(1).to_string())


if __name__ == '__main__':
    main()
