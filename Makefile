# Semua target memakai .venv lokal yang dibuat oleh `make setup`.
UV  ?= uv
PY  := $(shell cat .python-version)
BIN := .venv/bin
# torch/torchvision dari index CPU PyTorch (Linux: wheel +cpu tanpa paket CUDA; macOS: wheel biasa)
TORCH := --torch-backend cpu
CONFIG ?= configs/baseline.yaml
DOMAIN_IMG ?= train-data/domain-check/conveyor_good_01.webp
APP_CONFIG ?= configs/baseline.yaml
DEMO_VIDEO ?= samples/demo.mp4
CAMERA ?= 0

.PHONY: setup lock test lint format manifest audit train evaluate train-anomaly \
	evaluate-anomaly calibrate compare domain-check demo-video demo demo-live camera clean

setup:  ## buat .venv + pasang dependency terkunci + paket editable
	$(UV) venv --python $(PY) .venv
	$(UV) pip install --python $(BIN)/python $(TORCH) -r requirements-dev.txt
	$(UV) pip install --python $(BIN)/python --no-deps -e .

lock:  ## perbarui requirements*.txt terkunci (label lokal +cpu dibuang agar pin sama di macOS)
	$(UV) pip compile pyproject.toml --python-version $(PY) $(TORCH) --custom-compile-command "make lock" -o requirements.txt
	$(UV) pip compile pyproject.toml --python-version $(PY) $(TORCH) --custom-compile-command "make lock" --extra dev -o requirements-dev.txt
	sed -i.bak -E 's/^(torch(vision)?==[0-9.]+)\+cpu/\1/' requirements.txt requirements-dev.txt
	rm -f requirements.txt.bak requirements-dev.txt.bak

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

evaluate:  ## evaluasi scorer A → outputs/baseline/report.md + docs/results_m3_baseline.md
	$(BIN)/python -m loaflens.evaluate --config configs/baseline.yaml

train-anomaly:  ## latih scorer B (normal saja; unduh bobot ResNet18 sekali) → outputs/anomaly/
	$(BIN)/python -m loaflens.train --config configs/anomaly.yaml

evaluate-anomaly:  ## evaluasi scorer B (+ heatmap) → outputs/anomaly/ + docs/results_m3_anomaly.md
	$(BIN)/python -m loaflens.evaluate --config configs/anomaly.yaml

calibrate:  ## saran threshold dari CV train+val (bukan test) → outputs/<run>/calibration.yaml
	$(BIN)/python -m loaflens.calibrate --config $(CONFIG)

compare:  ## tabel A vs B dari metrics.json → outputs/compare.md + docs/results_m3.md
	$(BIN)/python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml

domain-check:  ## skor tiap roti di foto konveyor → outputs/<run>/domain_check.jpg
	$(BIN)/python -m loaflens.evaluate --config $(CONFIG) --domain-check "$(DOMAIN_IMG)"

demo-video:  ## video demo SYNTHETIC dari foto konveyor → samples/demo.mp4 + demo.gt.json
	$(BIN)/python -m loaflens.synth --config $(APP_CONFIG) --out $(DEMO_VIDEO)

demo:  ## app headless pada video demo + banding ground truth → outputs/demo_out.mp4, outputs/app/
	$(BIN)/python -m loaflens.app --config $(APP_CONFIG) --source video:$(DEMO_VIDEO) --headless \
		--out outputs/demo_out.mp4 --gt $(DEMO_VIDEO:.mp4=.gt.json)

demo-live:  ## app dengan jendela (q/ESC keluar) pada video demo
	$(BIN)/python -m loaflens.app --config $(APP_CONFIG) --source video:$(DEMO_VIDEO)

camera:  ## app dengan jendela dari kamera (CAMERA=0); demo kamera-ke-layar, lihat docs/USAGE.md
	$(BIN)/python -m loaflens.app --config $(APP_CONFIG) --source camera:$(CAMERA)

clean:
	rm -rf .venv .pytest_cache .ruff_cache
