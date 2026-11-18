.PHONY: help install test lint format demo clean

PYTHON ?= python

help:
	@echo "ALPR-Vietnamese Development Commands:"
	@echo "  make install    Install dependencies from requirements.txt"
	@echo "  make test       Run pytest test suite"
	@echo "  make lint       Run code linting with ruff"
	@echo "  make format     Format code with ruff"
	@echo "  make demo       Start web dashboard"
	@echo "  make clean      Remove caches and temporary files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check alpr/ scripts/ data_tools/ evaluation/ experiments/

format:
	$(PYTHON) -m ruff format alpr/ scripts/ data_tools/ evaluation/ experiments/

demo:
	$(PYTHON) scripts/stream_app.py --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
