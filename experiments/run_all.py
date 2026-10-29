"""
Executa a grade completa de experimentos (v2) em paralelo.

  Experimento 1 (principal): TBFL sem ataque, 100 rodadas, 5 seeds.
  Experimento 2 (Sybil): fração de Sybils x agregador, 50 rodadas, 3 seeds.
"""
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.environ.get('TBFL_OUT', os.path.join(HERE, '..', 'results'))
WORKERS = int(os.environ.get('TBFL_WORKERS', 8))

jobs = []
for seed in range(5):
    jobs.append(['--agg', 'tbfl', '--sybils', '0', '--seed', str(seed),
                 '--rounds', '100', '--per_client'])
for seed, s, agg in itertools.product(range(3), [2, 5, 10, 15],
                                      ['fedavg', 'median', 'trimmed', 'krum', 'tbfl']):
    jobs.append(['--agg', agg, '--sybils', str(s), '--seed', str(seed), '--rounds', '50'])
jobs.extend(['--agg', 'fedavg', '--sybils', '0', '--seed', str(seed), '--rounds', '50']
            for seed in range(3))

# Blocos de contas distintos para execuções TBFL simultâneas (Hardhat com 200 contas)
tbfl_slot = itertools.cycle(range(0, 196, 28))
for j in jobs:
    if 'tbfl' in j:
        j += ['--account_offset', str(next(tbfl_slot))]


def run(args):
    tag = '_'.join(args[1:6:2])
    log = open(os.path.join(OUT, 'logs', f'{tag}.log'), 'w')
    rc = subprocess.call([PY, os.path.join(HERE, 'run_experiment.py'), '--out', OUT] + args,
                         stdout=log, stderr=subprocess.STDOUT)
    return tag, rc


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
    # Pré-constrói as federações (uma por seed) antes da execução paralela
    sys.path.insert(0, HERE)
    from run_experiment import load_federation
    for seed in range(5):
        load_federation(os.path.join(HERE, '..', 'data', 'mortalidade_features.csv'), 10, 0.5, seed,
                        os.environ.get('TBFL_CACHE', os.path.join(HERE, '..', 'data', 'cache')))
        print(f'federação seed {seed} pronta', flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for tag, rc in ex.map(run, jobs):
            print(f'{tag}: {"ok" if rc == 0 else "ERRO " + str(rc)}', flush=True)
