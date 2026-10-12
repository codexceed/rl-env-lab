# Handles for human-controlled work. `make` alone lists them.
#
# Local dev runs anywhere; GPU targets run on the Lightning studio (Linux, `make setup-gpu` first).
# Variables can be overridden, e.g.: make train RUN=2b-leak-s2   make toy MODEL=Qwen/Qwen3.5-0.8B LR=3e-6

UV := uv run
PRECOMMIT := uvx --with pre-commit-uv pre-commit

MODEL ?= Qwen/Qwen3.5-2B
LR ?= 3e-6
LRS ?= 1e-6 3e-6 1e-5
STEPS ?= 100
RUN ?= 2b-rob-s1
STEP ?= final
NODE ?= node1
DATA ?= data
MACHINE ?= H200
CMD ?= nvidia-smi

# run id <size>-<rob|leak>-s<seed>  ->  size, reward, seed
SIZE = $(word 1,$(subst -, ,$(RUN)))
REWARD = $(if $(filter rob,$(word 2,$(subst -, ,$(RUN)))),robust,leaky)
SEED = $(patsubst s%,%,$(word 3,$(subst -, ,$(RUN))))

.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu hooks check fmt test lock data acceptance compat toy sweep bench \
	train eval eval-base diagnostic plan coordinator watcher report report-fake studio-status studio-switch studio-run

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

# --- setup and checks ---------------------------------------------------------------------------

setup: hooks  ## First-time local setup: dev env + pre-commit hooks
	uv sync --group dev

setup-gpu:  ## GPU node setup: dev + gpu extra, then build causal-conv1d (no PyPI wheels; needs CUDA 13 nvcc)
	uv sync --group dev --extra gpu
	uv pip install setuptools wheel ninja packaging
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$${MAX_JOBS:-16} \
		uv pip install --no-build-isolation --no-deps 'causal-conv1d==1.7.0'

hooks:  ## Install the pre-commit hooks
	$(PRECOMMIT) install

check:  ## Every lint, format, type, docstring and import check, as CI runs them
	$(PRECOMMIT) run --all-files

fmt:  ## Auto-fix lint and format
	$(UV) ruff check --fix .
	$(UV) ruff format .

test:  ## Unit tests (no GPU, no account, no downloads)
	$(UV) pytest -q

lock:  ## Re-resolve uv.lock after a dependency change
	uv lock

# --- environment ---------------------------------------------------------------------------------

data:  ## Build train / held-out / broken splits into DATA
	$(UV) python -m envlab.tasks.build --out $(DATA)

acceptance:  ## Quality gates G1-G3 on every instance; exits 1 if acceptance fails
	$(UV) python -m envlab.verify.acceptance --data $(DATA)

# --- GPU: checks and benchmarks (§0.3, §0.4) -----------------------------------------------------

compat:  ## Qwen3.5 full-training-path checks for MODEL -> runs/compat/
	$(UV) python train/compat_check.py --model $(MODEL)

toy:  ## Toy GRPO (addition) for MODEL at LR, STEPS steps -> runs/toy/
	$(UV) python train/smoke_toy.py --model $(MODEL) --lr $(LR) --steps $(STEPS)

sweep:  ## Toy GRPO at every LR in LRS for MODEL
	for lr in $(LRS); do $(MAKE) toy MODEL=$(MODEL) LR=$$lr STEPS=$(STEPS) || exit 1; done

bench:  ## Step time, peak memory and eval throughput at real task lengths
	$(UV) python train/bench.py

# --- GPU: training and eval ----------------------------------------------------------------------

train:  ## One GRPO run by id, e.g. RUN=2b-leak-s2 (outputs runs/<RUN>/)
	$(UV) python train/grpo.py --run-id $(RUN) --size $(SIZE) --reward $(REWARD) --seed $(SEED) \
		--output-dir runs/$(RUN)

eval:  ## Held-out eval of RUN at STEP (step number or `final`) -> evals/<RUN>/step-NNNNN.jsonl
	$(UV) python eval/heldout.py --run-id $(RUN) --size $(SIZE) --reward $(REWARD) --seed $(SEED) \
		--checkpoint runs/$(RUN) --step $(STEP)

eval-base:  ## Formal eval of the base model, e.g. SIZE=2b -> evals/base-<SIZE>/step-00000.jsonl
	$(UV) python eval/heldout.py --size $(SIZE) --checkpoint base

diagnostic:  ## Diagnostic baseline on train templates (band check), e.g. SIZE=2b -> evals/diagnostic/
	$(UV) python eval/heldout.py --size $(SIZE) --checkpoint base --mode diagnostic

plan:  ## Print what the coordinator would launch on NODE
	$(UV) python train/coordinator.py --node $(NODE) --print-plan

coordinator:  ## Launch NODE's runs + eval watcher per configs/allocation.yaml (never releases the node)
	$(UV) python train/coordinator.py --node $(NODE)

watcher:  ## Eval watcher alone for NODE
	$(UV) python eval/watcher.py --config configs/allocation.yaml --node $(NODE)

# --- results -------------------------------------------------------------------------------------

report:  ## Fill reports/results.md tables, verdicts and figures from evals/ and runs/
	$(UV) python eval/compare.py

report-fake:  ## The report pipeline end to end on generated fake data
	$(UV) python eval/fake_data.py --out $(TMPDIR)/envlab-fake && $(UV) python eval/compare.py --root $(TMPDIR)/envlab-fake

# --- Lightning studio (run from a laptop) --------------------------------------------------------

studio-status:  ## Studio status, machine and auto-sleep
	uv run --no-project --with lightning-sdk python scripts/studio.py status

studio-switch:  ## Switch the studio to MACHINE (e.g. H200, H200_X_8). Releasing GPUs is the owner's call
	uv run --no-project --with lightning-sdk python scripts/studio.py switch $(MACHINE)

studio-run:  ## Run CMD in ~/rl-env-lab on the studio
	uv run --no-project --with lightning-sdk python scripts/studio.py run "$(CMD)"
