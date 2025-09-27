# Finance Engine v1 Makefile

.PHONY: install run test lint clean

install:
	python3 -m pip install -r requirements.txt

run:
	python3 run.py

# Minimal smoke tests
test:
	pytest -q

# Placeholder lint (extend later)
lint:
	@echo "No linters configured yet."

clean:
	rm -rf outputs/*
