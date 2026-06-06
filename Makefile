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

# ElevenLabs voice pipeline knobs (override on the command line).
TEXT  ?= Officer, you have grounds to search under PACE section 1.
OUT   ?= data/audio/sample.mp3
AUDIO ?= $(OUT)
CONV   ?= data/audio/conversation.wav
REF    ?= data/audio/officer_ref.mp3
SNR    ?= 15
OTHERS ?= 3

# Video-to-text (Nebius Qwen2.5-VL) knobs.
VIDEO  ?= data/video/sample.mp4
FRAMES ?= 8
FPS    ?= 1

.PHONY: all data setup scrape corpus generate smoke validate split tts stt identify demo-audio video2text clean distclean help

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
	@echo "  tts       text -> speech via ElevenLabs (TEXT=... OUT=...)"
	@echo "  stt       speech -> text via ElevenLabs (AUDIO=...)"
	@echo "  demo-audio build a noisy multi-speaker demo (SNR=... OTHERS=1..3)"
	@echo "  identify  label officer vs person1/2/... in a conversation (CONV=... REF=...)"
	@echo "  video2text describe a video via Nebius Qwen2.5-VL (VIDEO=... FRAMES=... FPS=...)"
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

# ElevenLabs voice pipeline (needs ELEVENLABS_API_KEY in .env).
tts:
	$(PY) src/audio.py tts "$(TEXT)" --out "$(OUT)"

stt:
	$(PY) src/audio.py stt "$(AUDIO)"

demo-audio:
	$(PY) src/make_demo_audio.py --snr-db $(SNR) --num-others $(OTHERS) --out "$(CONV)" --ref "$(REF)"

identify:
	$(PY) src/speaker_id.py identify --conversation "$(CONV)" --officer "$(REF)"

video2text:
	$(PY) src/video_to_text.py "$(VIDEO)" --max-frames $(FRAMES) --fps $(FPS)

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
