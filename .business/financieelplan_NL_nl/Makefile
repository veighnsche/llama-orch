# Simple dev Makefile for finance engine v2 (single-input machine)

.PHONY: all check keys templates validate run migrate-v1 clean

# Canonical single input config for the deterministic engine
INPUT?=config.yaml

all: check templates

check:
	python3 tools/check_lines.py

keys:
	./fp list-keys

templates:
	./fp check-templates

validate:
	./fp validate $(INPUT) --months 12

run:
	./fp run $(INPUT) --out ./out --months 12 --scenario base --vat-period monthly

# Migrate a v1 input (omzetmodel/opex_pm) to v2 schema
# Writes to $(INPUT) by default for single-config workflows
OUT?=$(INPUT)
migrate-v1:
	python3 tools/migrate_v1_to_v2.py $(INPUT) $(OUT)

clean:
	rm -rf out
