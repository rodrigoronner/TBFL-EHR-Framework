#!/bin/bash
# Seeds 3 e 4 dos Experimentos 2 e 3 e parada antecipada (79 execuções). Requer o nó Hardhat.
#   nohup ./run_seeds.sh > seeds.out 2>&1 &
set -e
cd "$(dirname "$0")"
WORK="${TBFL_WORK:-$HOME/tbfl_work}"
# Espera a grade do passo 2 terminar, se estiver em execução
until grep -q '^fim' "$WORK/step2.out" 2>/dev/null; do sleep 60; done
echo "inicio $(date +%H:%M)"
TBFL_CACHE="$WORK/cache" TBFL_WORKERS=10 "$WORK/venv/bin/python" run_all.py seeds > "$WORK/run_seeds.log" 2>&1
echo "fim $(date +%H:%M): $(grep -c ': ok' "$WORK/run_seeds.log") ok, $(grep -c ERRO "$WORK/run_seeds.log" || true) erros"
