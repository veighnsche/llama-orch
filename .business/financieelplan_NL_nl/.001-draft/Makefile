# Simple dev Makefile for finance engine v2 (single-input machine)

.PHONY: all check keys templates validate run migrate-v1 clean compute-costs

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

# Compute research-based GPU/model costs and write 15_compute_costs.{md,csv}
compute-costs:
	python3 tools/compute_costs.py

# Full pipeline: compute research costs then render reports (ensures 15_*.md is included in combined doc)
run-all: compute-costs run

# Migrate a v1 input (omzetmodel/opex_pm) to v2 schema
# Writes to $(INPUT) by default for single-config workflows
OUT?=$(INPUT)
migrate-v1:
	python3 tools/migrate_v1_to_v2.py $(INPUT) $(OUT)

clean:
	rm -rf out
