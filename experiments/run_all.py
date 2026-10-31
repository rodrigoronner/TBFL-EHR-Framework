"""
Executa a grade completa de experimentos (v2) em paralelo.

  Experimento 1 (principal): TBFL sem ataque, 100 rodadas, 5 seeds.
  Experimento 2 (Sybil): fração de Sybils x agregador, 50 rodadas, 3 seeds.
  Experimento 3 (`python run_all.py extra`): FoolsGold e ataque adaptativo, 50 rodadas, 3 seeds.
  Experimento 4 (`python run_all.py step2`): gate + mediana, varredura de sigma, norm clipping e
  Sybils com rótulos invertidos, 50 rodadas, 3 seeds.
  Revisão 2 (`python run_all.py rev2`): média aparada com rótulos invertidos, gate com S=15
  (adaptativo e rótulos invertidos) e insiders credenciados, 50 rodadas, 3 seeds.
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
R50 = ['--rounds', '50']
if 'seeds' in sys.argv[1:]:
    # Seeds 3 e 4 para os Experimentos 2 e 3 (total de cinco seeds) e parada antecipada
    for seed, s, agg in itertools.product([3, 4], [2, 5, 10, 15],
                                          ['fedavg', 'median', 'trimmed', 'krum', 'tbfl',
                                           'foolsgold']):
        jobs.append(['--agg', agg, '--sybils', str(s), '--seed', str(seed)] + R50)
    for seed, s, agg in itertools.product([3, 4], [5, 10],
                                          ['foolsgold', 'median', 'trimmed', 'krum']):
        jobs.append(['--agg', agg, '--sybils', str(s), '--seed', str(seed),
                     '--sybil_diversity', '1'] + R50)
    for seed, agg in itertools.product([3, 4], ['fedavg', 'median', 'trimmed', 'krum',
                                               'foolsgold']):
        jobs.append(['--agg', agg, '--sybils', '0', '--seed', str(seed)] + R50)
    # Parada antecipada: FedAvg sem ataque (equivalente ao TBFL sem Sybils), 100 rodadas,
    # rodada escolhida pela AUC de validação, separada do teste
    for seed in range(5):
        jobs.append(['--agg', 'fedavg', '--sybils', '0', '--seed', str(seed), '--rounds', '100',
                     '--val_frac', '0.1'])
elif 'step2' in sys.argv[1:]:
    other, gated = [], []
    # Gate combinado com a mediana sob Sybils idênticos
    for seed, s in itertools.product(range(3), [5, 10, 15]):
        gated.append(['--agg', 'tbfl', '--rule', 'median', '--sybils', str(s), '--seed', str(seed)])
    # Varredura da diversidade do ataque adaptativo com maioria de Sybils
    for seed, sg, agg in itertools.product(range(3), ['0.1', '0.3'],
                                           ['median', 'krum', 'foolsgold']):
        other.append(['--agg', agg, '--sybils', '10', '--seed', str(seed),
                      '--sybil_diversity', sg])
    # Norm clipping sem ataque e sob Sybils idênticos
    for seed, s in itertools.product(range(3), [0, 2, 5, 10, 15]):
        other.append(['--agg', 'normclip', '--sybils', str(s), '--seed', str(seed)])
    # Sybils plausíveis: rótulos invertidos
    for seed, s, agg in itertools.product(range(3), [5, 10],
                                          ['fedavg', 'median', 'krum', 'foolsgold', 'normclip',
                                           'tbfl']):
        (gated if agg == 'tbfl' else other).append(
            ['--agg', agg, '--sybils', str(s), '--seed', str(seed), '--sybil_mode', 'labelflip'])
    # Intercala as execuções com cadeia para não esgotar os blocos de contas do nó
    step = max(len(other) // max(len(gated), 1), 1)
    for i, j in enumerate(other):
        if i % step == 0 and gated:
            jobs.append(gated.pop(0) + R50)
        jobs.append(j + R50)
    jobs.extend(g + R50 for g in gated)
elif 'rev2' in sys.argv[1:]:
    # Resposta ao segundo parecer: média aparada com rótulos invertidos; gate com S=15 nos
    # ataques adaptativo e de rótulos invertidos; insiders credenciados sob gate + FedAvg/mediana
    for seed, s in itertools.product(range(3), [5, 10]):
        jobs.append(['--agg', 'trimmed', '--sybils', str(s), '--seed', str(seed),
                     '--sybil_mode', 'labelflip'] + R50)
    for seed in range(3):
        jobs.append(['--agg', 'tbfl', '--sybils', '15', '--seed', str(seed),
                     '--sybil_diversity', '1'] + R50)
        jobs.append(['--agg', 'tbfl', '--sybils', '15', '--seed', str(seed),
                     '--sybil_mode', 'labelflip'] + R50)
    for seed, k, rule in itertools.product(range(3), [1, 2], ['fedavg', 'median']):
        jobs.append(['--agg', 'tbfl', '--rule', rule, '--sybils', '0', '--insiders', str(k),
                     '--seed', str(seed)] + R50)
elif 'extra' in sys.argv[1:]:
    # Experimento 3: FoolsGold sob Sybils idênticos e ataque adaptativo (Sybils diversificados)
    for seed, s in itertools.product(range(3), [2, 5, 10, 15]):
        jobs.append(['--agg', 'foolsgold', '--sybils', str(s), '--seed', str(seed),
                     '--rounds', '50'])
    for seed, s, agg in itertools.product(range(3), [5, 10],
                                          ['foolsgold', 'median', 'trimmed', 'krum']):
        jobs.append(['--agg', agg, '--sybils', str(s), '--seed', str(seed), '--rounds', '50',
                     '--sybil_diversity', '1'])
    # Agregadores robustos sem ataque: equivalem a gate + agregador robusto sob qualquer
    # fração de Sybils, pois o gate impede que identidades não credenciadas cheguem ao agregador
    for seed, agg in itertools.product(range(3), ['median', 'trimmed', 'krum', 'foolsgold']):
        jobs.append(['--agg', agg, '--sybils', '0', '--seed', str(seed), '--rounds', '50'])
else:
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
    kv = dict(zip(args[::2], args[1::2]))
    kv.pop('--account_offset', None)
    tag = '_'.join(v for k, v in kv.items() if k not in ('--rounds',))
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
        if 'seeds' in sys.argv[1:]:
            load_federation(os.path.join(HERE, '..', 'data', 'mortalidade_features.csv'), 10, 0.5,
                            seed, os.environ.get('TBFL_CACHE',
                                                 os.path.join(HERE, '..', 'data', 'cache')), 0.1)
        print(f'federação seed {seed} pronta', flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for tag, rc in ex.map(run, jobs):
            print(f'{tag}: {"ok" if rc == 0 else "ERRO " + str(rc)}', flush=True)
