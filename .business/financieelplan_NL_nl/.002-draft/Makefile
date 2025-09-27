# Finance Engine v1 Makefile

.PHONY: install setup run test lint clean js-install pdf

setup:
	python3 -m venv .venv
	./.venv/bin/pip install -U pip
	./.venv/bin/pip install -r requirements.txt

install: setup

run:
	./.venv/bin/python run.py

# Minimal smoke tests
test:
	./.venv/bin/pytest -q

# Placeholder lint (extend later)
lint:
	@echo "No linters configured yet."

clean:
	rm -rf outputs/*

js-install:
	npm install

# Generate PDF that includes Mermaid by pre-rendering diagrams in outputs/financial_plan.md
pdf: run js-install
	./.venv/bin/python scripts/prerender_mermaid.py
	npm run pdf
