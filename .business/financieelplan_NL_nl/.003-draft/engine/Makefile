PY=python3
PKG:=$(CURDIR)
VENV?=$(PKG)/.venv

.PHONY: help venv install run test fmt lint audit

help:
	@echo "Targets: venv, install, run, test, fmt, lint, audit"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel

install: venv
	$(VENV)/bin/pip install -r $(PKG)/requirements.txt

run:
	$(VENV)/bin/python -m d3_engine.cli --inputs ../inputs --out ../outputs --pipelines public,private --seed 424242

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
