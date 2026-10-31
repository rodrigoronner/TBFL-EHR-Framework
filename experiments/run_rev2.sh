#!/bin/bash
# Revisão 2: 24 execuções de 50 rodadas. Requer um nó Hardhat em 127.0.0.1:8545.
# No máximo 7 execuções com cadeia ao mesmo tempo (7 blocos de 28 contas).
set -e
cd "$(dirname "$0")"
WORK="${TBFL_WORK:-$HOME/tbfl_work}"
echo "inicio $(date +%H:%M)"
TBFL_CACHE="$WORK/cache" TBFL_WORKERS=7 "$WORK/venv/bin/python" run_all.py rev2 > "$WORK/run_rev2.log" 2>&1
echo "fim $(date +%H:%M): $(grep -c ': ok' "$WORK/run_rev2.log") ok, $(grep -c ERRO "$WORK/run_rev2.log" || true) erros"
