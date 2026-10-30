#!/bin/bash
# Retoma o Experimento 3 (FoolsGold, ataque adaptativo e agregadores robustos sem ataque)
# e regenera figuras e report.json. Roda desvinculado do terminal:
#   nohup ./retomar_grade_extra.sh > retomada.out 2>&1 &
# Duração estimada: ~5 h em 10 núcleos (48 execuções de 50 rodadas).
set -e
cd "$(dirname "$0")"
WORK="${TBFL_WORK:-$HOME/tbfl_work}"          # venv e cache fora do OneDrive e fora de /tmp
mkdir -p "$WORK"

if [ ! -x "$WORK/venv/bin/python" ]; then
  # Mesmo interpretador e mesmas versões da grade principal (requirements.txt do repositório)
  python3.13 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q torch==2.15.0 numpy==2.5.4 pandas==3.1.0 scikit-learn==1.9.1 \
    imbalanced-learn==0.14.2 scipy==1.18.1 matplotlib==3.11.2 web3==8.0.0
fi

echo "inicio $(date +%H:%M)"
TBFL_CACHE="$WORK/cache" TBFL_WORKERS=10 "$WORK/venv/bin/python" run_all.py extra > "$WORK/run_extra.log" 2>&1
echo "grade extra fim $(date +%H:%M): $(grep -c ': ok' "$WORK/run_extra.log") ok, $(grep -c ERRO "$WORK/run_extra.log" || true) erros"
TBFL_CACHE="$WORK/cache" "$WORK/venv/bin/python" make_figures.py > "$WORK/report_final.txt" 2>&1
echo "figuras fim $(date +%H:%M)"
