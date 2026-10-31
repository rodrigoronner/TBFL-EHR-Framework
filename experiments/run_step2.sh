#!/bin/bash
# Experimento 4 (resposta ao parecer): gate + mediana, varredura de sigma, norm clipping e
# Sybils com rótulos invertidos. 78 execuções de 50 rodadas. Requer um nó Hardhat em
# 127.0.0.1:8545 (npx hardhat node). Roda desvinculado do terminal:
#   nohup ./run_step2.sh > step2.out 2>&1 &
set -e
cd "$(dirname "$0")"
WORK="${TBFL_WORK:-$HOME/tbfl_work}"
echo "inicio $(date +%H:%M)"
TBFL_CACHE="$WORK/cache" TBFL_WORKERS=10 "$WORK/venv/bin/python" run_all.py step2 > "$WORK/run_step2.log" 2>&1
echo "fim $(date +%H:%M): $(grep -c ': ok' "$WORK/run_step2.log") ok, $(grep -c ERRO "$WORK/run_step2.log" || true) erros"
