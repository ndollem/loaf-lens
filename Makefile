# Semua target memakai .venv lokal yang dibuat oleh `make setup`.
UV  ?= uv
PY  := $(shell cat .python-version)
BIN := .venv/bin

.PHONY: setup lock test lint format manifest audit train evaluate clean

setup:  ## buat .venv + pasang dependency terkunci + paket editable
	$(UV) venv --python $(PY) .venv
	$(UV) pip install --python $(BIN)/python -r requirements-dev.txt
	$(UV) pip install --python $(BIN)/python --no-deps -e .

lock:  ## perbarui requirements*.txt terkunci dari pyproject.toml
	$(UV) pip compile pyproject.toml --python-version $(PY) -o requirements.txt
	$(UV) pip compile pyproject.toml --python-version $(PY) --extra dev -o requirements-dev.txt

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

manifest:  ## bangun train-data/manifest.csv (tidak menimpa; split ulang: tambah --force manual)
	$(BIN)/python -m loaflens.data --config configs/baseline.yaml

audit:  ## laporan audit dataset → outputs/audit.md + docs/audit_m1.md
	$(BIN)/python -m loaflens.audit --config configs/baseline.yaml

train:  ## latih scorer A (baseline) dari split train → outputs/baseline/
	$(BIN)/python -m loaflens.train --config configs/baseline.yaml

evaluate:  ## evaluasi val/test + CV train+val → outputs/baseline/report.md + docs/results_m2.md
	$(BIN)/python -m loaflens.evaluate --config configs/baseline.yaml

clean:
	rm -rf .venv .pytest_cache .ruff_cache
