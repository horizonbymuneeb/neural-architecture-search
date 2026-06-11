PYTHON ?= python3
PIP := $(PYTHON) -m pip
export PYTHONPATH := src

.PHONY: install test lint space data search compare clean

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src

space:
	$(PYTHON) -m nas space

data:
	$(PYTHON) -m nas data

search:
	$(PYTHON) -m nas search --strategy both

compare:
	$(PYTHON) -m nas search --strategy both --json --out artifacts/comparison.json

clean:
	rm -rf artifacts .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
