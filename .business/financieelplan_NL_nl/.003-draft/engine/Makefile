PY=python3
PKG:=$(CURDIR)
VENV?=$(PKG)/.venv

.PHONY: venv install run test

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel

install: venv
	$(VENV)/bin/pip install -r $(PKG)/requirements.txt

run:
	$(VENV)/bin/python -m d3_engine.cli --inputs ../inputs --out ../outputs --pipelines public,private --seed 424242

test:
	$(VENV)/bin/python -m pytest -q $(PKG)/tests
