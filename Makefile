# Hardware decides two things before anything is installed: which torch build
# matches the driver, and which preset fits the card. Both are overridable.
DETECTED    := $(shell sh scripts/detect.sh 2>/dev/null)
AUTO_CUDA   := $(patsubst TORCH_CUDA=%,%,$(filter TORCH_CUDA=%,$(DETECTED)))
AUTO_PRESET := $(patsubst PRESET=%,%,$(filter PRESET=%,$(DETECTED)))
AUTO_GPUS   := $(patsubst GPU_COUNT=%,%,$(filter GPU_COUNT=%,$(DETECTED)))

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
# GPU=1 picks the second card; GPU=0,2 hands train-multi exactly those two.
# DEVICE=cuda:1 is the long form of a single card. CUDA=cu121 forces a torch
# build. `make detect` lists the cards with their indices.
# Trailing comments are kept off these lines on purpose: make would take the
# spaces before the "#" as part of the value, and an "empty" variable holding a
# space is treated as set.
GPU    ?=
COMMA  := ,
# These are immediate assignments, so GPU and COMMA have to exist by now.
GPU_LIST := $(subst $(COMMA), ,$(strip $(GPU)))
MANY := $(word 2,$(GPU_LIST))

# A list of cards goes through CUDA_VISIBLE_DEVICES, and each process then
# takes cuda:0, cuda:1 ... of what it can see - so run.device must stay unset.
DEVICE ?= $(if $(MANY),,$(if $(strip $(GPU)),cuda:$(strip $(GPU))))
VISIBLE := $(if $(MANY),CUDA_VISIBLE_DEVICES=$(strip $(GPU)))
# How many processes to train with. A list of cards decides it; GPUS=all takes
# every card the machine has; otherwise one.
GPUS   ?= $(if $(MANY),$(words $(GPU_LIST)),1)
# "0" is what detect prints on a machine with no card, and it is not empty, so
# it has to be filtered out or torchrun is asked for zero processes.
COUNTED := $(strip $(filter-out 0,$(AUTO_GPUS)))
WANTED := $(strip $(if $(filter all,$(strip $(GPUS))),$(if $(COUNTED),$(COUNTED),1),$(GPUS)))
# More than one process means torchrun, and that is the only difference: every
# target below launches the same way, so `make all GPUS=2` needs nothing else.
# --standalone: one machine, rendezvous on localhost and a free port. Without
# it torchrun resolves the host name, which on some machines points outward
# and leaves the processes unable to reach each other.
LAUNCH := $(if $(filter-out 1,$(WANTED)),\
            $(VENV)/bin/torchrun --standalone --nproc_per_node=$(WANTED),$(PY))
CUDA   ?= $(AUTO_CUDA)
ARGS   ?=

# Training stops once held-out perplexity is falling slower than STOP points
# an hour. Left empty it is 0.05, the value in config.toml - not the whole
# token budget, so STOP=0 is how a run is made to use all of it. WINDOW is how
# many hours the rate is measured over and PATIENCE how many checks in a row
# must be slow, so one noisy measurement cannot end a run.
STOP     ?=
WINDOW   ?=
PATIENCE ?=
STOP_ARGS := $(if $(strip $(STOP)),--set train.min_improvement_per_hour=$(strip $(STOP))) \
             $(if $(strip $(WINDOW)),--set train.improvement_window_hours=$(strip $(WINDOW))) \
             $(if $(strip $(PATIENCE)),--set train.stop_patience=$(strip $(PATIENCE)))

# DATA is olmo, fineweb, dolma or text. SUBSETS names domains, empty takes all.
# olmo by default: dolma is served from olmo-data.org alone, which some
# networks cannot reach, while the same corpus family sits on the hub as
# allenai/olmo-mix-1124. dclm is the web part and nearly all of the text.
DATA    ?= olmo
SUBSETS ?= ["dclm","pes2o","wiki","arxiv"]
DATA_ARGS := --set data.kind=$(strip $(DATA)) \
             $(if $(strip $(SUBSETS)),--set 'data.subsets=$(strip $(SUBSETS))')
FLAGS  := --preset $(CHOSEN) \
          $(if $(strip $(DEVICE)),--set run.device=$(strip $(DEVICE))) \
          $(STOP_ARGS) $(ARGS)

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
	$(VISIBLE) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	$(LAUNCH) main.py train $(FLAGS) $(DATA_ARGS)

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

# Kept as a name people already type; `train` does the same thing when it is
# given more than one card, including inside `make all`.
train-multi: train

# Prove the folder works with the network unplugged.
offline:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PY) main.py status $(FLAGS)

clean:
	rm -rf $(VENV) __pycache__ distill/__pycache__ *.pyc
