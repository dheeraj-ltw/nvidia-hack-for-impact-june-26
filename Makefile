# PoliceAI (England & Wales) fine-tuning dataset pipeline.
# Final training data is written to ./training_data/ in format.jsonl shape.
#
# Quick start:
#   make setup          # create venv + install deps
#   cp .env.example .env && edit .env   # add required keys
#   make data           # scrape -> corpus -> generate -> validate -> split
#
# Generate a different size:   make data N=200
# Smoke test (5 examples):     make smoke

PY := .venv/bin/python
N  ?= 100

.PHONY: all data setup scrape corpus generate smoke validate split clean distclean help

help:
	@echo "Targets:"
	@echo "  setup     create .venv and install requirements"
	@echo "  scrape    scrape statutes + case law + police guidance"
	@echo "  corpus    build the citation-tagged corpus index"
	@echo "  smoke     generate 5 examples (needs ANTHROPIC_API_KEY)"
	@echo "  generate  generate N=$(N) examples (needs ANTHROPIC_API_KEY)"
	@echo "  validate  schema / jurisdiction / citation / dedupe checks"
	@echo "  split     publish train/val + dataset.jsonl to training_data/"
	@echo "  data      run the full pipeline (scrape..split) with N=$(N)"
	@echo "  clean     remove generated data (keeps cached scrapes + venv)"
	@echo "  distclean remove venv, caches and all generated data"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "Setup done. Now: cp .env.example .env  and add ANTHROPIC_API_KEY"

scrape:
	$(PY) src/scrape_legislation.py
	$(PY) src/scrape_caselaw.py
	$(PY) src/scrape_police_guidance.py

corpus:
	$(PY) src/build_corpus.py

smoke:
	$(PY) src/generate_dataset.py --n 5

generate:
	$(PY) src/generate_dataset.py --n $(N)

validate:
	$(PY) src/validate_dataset.py

split:
	$(PY) src/split_dataset.py

# Full end-to-end run.
data: scrape corpus generate validate split
	@echo "Training data ready in ./training_data/ (dataset.jsonl, train.jsonl, val.jsonl)"

all: data

clean:
	rm -rf data/output training_data data/corpus_index.jsonl
	find src -name '__pycache__' -type d -exec rm -rf {} +

distclean: clean
	rm -rf .venv data/corpus/legislation/*.xml data/corpus/caselaw/*.xml \
	       data/corpus/police_guidance/*.html
