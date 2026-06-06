#!/usr/bin/env bash
# End-to-end pipeline for the PoliceAI (England & Wales) fine-tuning dataset.
# Usage: ./run_all.sh [N]      (N = number of examples, default 100)
set -euo pipefail

N="${1:-100}"
PY=".venv/bin/python"
cd "$(dirname "$0")"

echo "== 1/6 scrape statutes (legislation.gov.uk) =="
$PY src/scrape_legislation.py
echo "== 2/6 scrape case law (Find Case Law) =="
$PY src/scrape_caselaw.py
echo "== 3/6 scrape police guidance =="
$PY src/scrape_police_guidance.py
echo "== 4/6 build corpus index =="
$PY src/build_corpus.py
echo "== 5/6 generate dataset (N=$N) [requires ANTHROPIC_API_KEY] =="
$PY src/generate_dataset.py --n "$N"
echo "== 6/6 validate + split =="
$PY src/validate_dataset.py
$PY src/split_dataset.py
echo "Done. Training data in ./training_data/{dataset.jsonl,train.jsonl,val.jsonl}"
