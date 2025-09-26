# Simple dev Makefile for finance engine v2

.PHONY: all check keys templates validate run clean

all: check templates

check:
	python3 tools/check_lines.py

keys:
	./fp list-keys

templates:
	./fp check-templates

validate:
	./fp validate examples/voorbeeld.json --months 12

run:
	./fp run examples/voorbeeld.json --out ./out --months 12 --scenario base --vat-period monthly

clean:
	rm -rf out
