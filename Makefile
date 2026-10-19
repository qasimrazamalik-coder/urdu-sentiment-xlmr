PY ?= python

# Used by `make data`, e.g. make data HF_ID=owner/name TEXT_COL=text LABEL_COL=label
HF_ID ?=
TEXT_COL ?= text
LABEL_COL ?= label

.PHONY: install lint format test smoke data train eval app clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

# End-to-end run on a tiny synthetic fixture. Proves the pipeline works; numbers are not results.
smoke:
	$(PY) -m urdu_sentiment.train --config configs/smoke.yaml
	$(PY) -m urdu_sentiment.evaluate --config configs/smoke.yaml
	$(PY) -m urdu_sentiment.error_analysis --config configs/smoke.yaml

data:
	$(PY) scripts/download_data.py --hf-id $(HF_ID) --text-col $(TEXT_COL) --label-col $(LABEL_COL)

train:
	$(PY) -m urdu_sentiment.train --config configs/default.yaml

eval:
	$(PY) -m urdu_sentiment.evaluate --config configs/default.yaml
	$(PY) -m urdu_sentiment.error_analysis --config configs/default.yaml

app:
	$(PY) app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
