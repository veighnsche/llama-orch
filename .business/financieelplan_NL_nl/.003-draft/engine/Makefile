PY=python3
PKG:=$(CURDIR)
VENV?=$(PKG)/.venv

# Run-time defaults (override on command line):
#   make run INPUTS=../inputs OUT=../outputs PIPELINES=public,private SEED=424242 FAIL_FLAGS=
INPUTS?=/home/vince/Projects/llama-orch/.business/financieelplan_NL_nl/.003-draft/inputs
OUT?=/home/vince/Projects/llama-orch/.business/financieelplan_NL_nl/.003-draft/outputs
PIPELINES?=public,private
SEED?=424242
# To disable fail-on-warning: make run FAIL_FLAGS=
FAIL_FLAGS?=--fail-on-warning

.PHONY: help venv install run quick test fmt lint audit

help:
	@echo "Targets: venv, install, run, quick, test, fmt, lint, audit"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel

install: venv
	$(VENV)/bin/pip install -r $(PKG)/requirements.txt

run:
	$(VENV)/bin/python -m cli --inputs $(INPUTS) --out $(OUT) --pipelines $(PIPELINES) --seed $(SEED) $(FAIL_FLAGS)

# Quick run using system Python (no venv)
quick:
	$(PY) -m cli --inputs $(INPUTS) --out $(OUT) --pipelines $(PIPELINES) --seed $(SEED) $(FAIL_FLAGS)

test:
	$(VENV)/bin/python -m pytest -q $(PKG)/tests

fmt:
	$(VENV)/bin/python -m pip install black
	$(VENV)/bin/black src tests

lint:
	$(VENV)/bin/python -m pip install ruff
	$(VENV)/bin/ruff check src

audit:
	$(VENV)/bin/python -m pip install pip-audit
	$(VENV)/bin/pip-audit -r $(PKG)/requirements.txt || true
