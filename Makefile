# Semua target memakai .venv lokal yang dibuat oleh `make setup`.
UV  ?= uv
PY  := $(shell cat .python-version)
BIN := .venv/bin

.PHONY: setup lock test lint format clean

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

clean:
	rm -rf .venv .pytest_cache .ruff_cache
