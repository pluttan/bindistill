VENV   := venv
PY     := $(VENV)/bin/python3.12
PIP    := $(VENV)/bin/pip

PRESET ?= small
GPUS   ?= 1
DEVICE ?=              # e.g. DEVICE=cuda:1 to pick a card
ARGS   ?=
FLAGS  := --preset $(PRESET) $(if $(DEVICE),--set run.device=$(DEVICE)) $(ARGS)

.PHONY: all install selftest status fetch train resume eval export smoke train-multi offline clean

all: install selftest

install:
	python3.12 -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

# Everything that can be checked without a download. Run this first.
selftest:
	$(PY) main.py selftest

status:
	$(PY) main.py status $(FLAGS)

# Needs a network. Puts the teacher and the corpus inside this folder.
fetch:
	$(PY) main.py fetch $(FLAGS)

train:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) main.py train $(FLAGS)

# Same as train; it always continues from the newest checkpoint.
resume: train

eval:
	$(PY) main.py eval $(FLAGS)

export:
	$(PY) main.py export $(FLAGS)

# Full cycle on a small model: a rehearsal that the real run will work.
smoke:
	$(MAKE) selftest
	$(PY) main.py fetch --preset smoke
	$(PY) main.py train --preset smoke
	$(PY) main.py eval --preset smoke

# Several cards on one machine.
train-multi:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	$(VENV)/bin/torchrun --nproc_per_node=$(GPUS) main.py train $(FLAGS)

# Prove the folder works with the network unplugged.
offline:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PY) main.py status $(FLAGS)

clean:
	rm -rf $(VENV) __pycache__ distill/__pycache__ *.pyc
