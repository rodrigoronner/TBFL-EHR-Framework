"""
Verifica, sem depender de exploradores, que os contratos implantados nas testnets públicas
foram criados exatamente a partir do bytecode publicado neste repositório.

Para cada implantação registrada em results/testnet_<rede>.csv:
  1. busca na rede a transação de criação e o seu recibo;
  2. confere que os dados da transação são o bytecode de experiments/<Contrato>.json seguido
     dos argumentos do construtor (o endereço do emissor, no FLRegistryZK);
  3. confere que o recibo criou o endereço listado em results/testnet_<rede>_meta.json e que
     há código implantado nele.

O bytecode dos artefatos pode ser reproduzido a partir de contracts/ com `npx hardhat compile`
(Solidity 0.8.24, configuração padrão do hardhat.config.js).

  python verify_deployments.py
"""
import json
import os

import pandas as pd
from eth_abi import encode
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
RPC = {'sepolia': 'https://ethereum-sepolia-rpc.publicnode.com',
       'arbsepolia': 'https://sepolia-rollup.arbitrum.io/rpc'}


def main():
    ok_all = True
    for net, url in RPC.items():
        meta_path = os.path.join(RES, f'testnet_{net}_meta.json')
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        log = pd.read_csv(os.path.join(RES, f'testnet_{net}.csv'))
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 60}))
        for _, row in log[log.op == 'deploy'].iterrows():
            name = row['contract']
            art = json.load(open(os.path.join(HERE, f'{name}.json')))
            expected = bytes.fromhex(art['bytecode'].removeprefix('0x'))
            if name == 'FLRegistryZK':
                expected += encode(['address'], [meta['issuer']])
            h = '0x' + str(row['tx']).removeprefix('0x')
            tx, rcpt = w3.eth.get_transaction(h), w3.eth.get_transaction_receipt(h)
            checks = {
                'creation code matches artifact': bytes(tx['input']) == expected,
                'created the listed address': rcpt['contractAddress'] == meta[name],
                'code present at address': len(w3.eth.get_code(meta[name])) > 0,
                'sent by the issuer': tx['from'] == meta['issuer'],
            }
            ok = all(checks.values())
            ok_all &= ok
            print(f'{net:10s} {name:13s} {meta[name]}  {"OK" if ok else "FALHOU"}')
            for k, v in checks.items():
                if not v:
                    print(f'    falhou: {k}')
    raise SystemExit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
