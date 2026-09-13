# Hardware decides two things before anything is installed: which torch build
# matches the driver, and which preset fits the card. Both are overridable.
DETECTED    := $(shell sh scripts/detect.sh 2>/dev/null)
AUTO_CUDA   := $(patsubst TORCH_CUDA=%,%,$(filter TORCH_CUDA=%,$(DETECTED)))
AUTO_PRESET := $(patsubst PRESET=%,%,$(filter PRESET=%,$(DETECTED)))

VENV   := venv
# The environment is built with python3.12, but the interpreter inside it is
# addressed without a version: a venv made by another 3.x still works, and every
# target keeps running instead of failing on a missing file name.
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# small by default; PRESET=auto sizes it from the card's memory instead.
PRESET ?= small
AUTO_OR_SMALL := $(if $(AUTO_PRESET),$(AUTO_PRESET),small)
CHOSEN := $(if $(filter auto,$(strip $(PRESET))),$(AUTO_OR_SMALL),$(strip $(PRESET)))
GPUS   ?= 1
# GPU=1 picks the second card, DEVICE=cuda:1 is the long form of the same.
# CUDA=cu121 forces a torch build. `make detect` lists the cards and indices.
# Trailing comments are kept off these lines on purpose: make would take the
# spaces before the "#" as part of the value, and an "empty" variable holding a
# space is treated as set.
GPU    ?=
DEVICE ?= $(if $(strip $(GPU)),cuda:$(strip $(GPU)))
CUDA   ?= $(AUTO_CUDA)
ARGS   ?=

# DATA is fineweb, dolma or text. SUBSETS names dolma domains; empty takes all.
DATA    ?= dolma
SUBSETS ?= ["books","c4-filtered","pes2o"]
DATA_ARGS := --set data.kind=$(strip $(DATA)) \
             $(if $(strip $(SUBSETS)),--set 'data.subsets=$(strip $(SUBSETS))')
FLAGS  := --preset $(CHOSEN) \
          $(if $(strip $(DEVICE)),--set run.device=$(strip $(DEVICE))) $(ARGS)

TORCH_INDEX := $(if $(strip $(CUDA)),\
                 --index-url https://download.pytorch.org/whl/$(strip $(CUDA)))

.PHONY: all update detect ensure install torch selftest status fetch train resume eval profile export smoke train-multi offline clean

# The whole thing: pick up the latest code, look at the hardware, install what
# matches it, check the machinery, fetch a mixture of domains, and train. Safe
# to re-run — every step continues rather than starts over.
all: update ensure detect selftest fetch train

# Latest code, but never at the cost of local work: a fast-forward or nothing.
update:
	@if git rev-parse --git-dir >/dev/null 2>&1; then \
		git pull --ff-only || echo "  pull skipped: local commits or changes, resolve by hand"; \
	else \
		echo "  not a git checkout, skipping update"; \
	fi

detect:
	@sh scripts/detect.sh --report

# Build the environment only when it is not there yet.
ensure:
	@if [ -x "$(PY)" ]; then \
		echo "  environment present"; \
	else \
		$(MAKE) install; \
	fi

install:
	python3.12 -m venv $(VENV) && $(PIP) install -U pip
	$(PIP) install $(TORCH_INDEX) torch
	$(PIP) install -r requirements.txt

# Swap the torch build without rebuilding the environment. Use when the driver
# is older than the default wheel expects.
torch:
	$(PIP) install --force-reinstall $(TORCH_INDEX) torch
	$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

# Everything that can be checked without a download. Run this first.
selftest:
	$(PY) main.py selftest

status:
	$(PY) main.py status $(FLAGS)

# Needs a network. Puts the teacher and the corpus inside this folder.
fetch:
	$(PY) main.py fetch $(FLAGS) $(DATA_ARGS)

train:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) main.py train $(FLAGS) $(DATA_ARGS)

# Same as train; it always continues from the newest checkpoint.
resume: train

eval:
	$(PY) main.py eval $(FLAGS) $(DATA_ARGS)

# Does our own result carry the signature we used to argue Bonsai was trained?
profile:
	$(PY) main.py profile $(FLAGS)

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
