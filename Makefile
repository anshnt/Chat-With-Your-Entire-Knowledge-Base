.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install test test-cov lint fmt typecheck check serve clean demo eval

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	python3 -m venv .venv

install: venv ## Install the project with dev extras
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

test: ## Run the test suite (offline, no API keys needed)
	$(PY) -m pytest

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov --cov-report=term-missing --cov-report=xml

lint: ## Lint with ruff
	.venv/bin/ruff check backend tests

fmt: ## Auto-fix lint findings and format
	.venv/bin/ruff check backend tests --fix
	.venv/bin/ruff format backend tests

typecheck: ## Type-check with mypy
	.venv/bin/mypy

check: lint typecheck test ## Lint, type-check, and test

serve: ## Run the API with auto-reload
	.venv/bin/kb serve --reload

demo: ## Ingest this repo's own docs and run a sample search
	.venv/bin/kb ingest ./docs ./README.md
	.venv/bin/kb stats
	.venv/bin/kb search "how does reciprocal rank fusion combine rankings?"

eval: ## Ingest this repo's docs and run the paraphrase retrieval sweep
	.venv/bin/kb --data-dir /tmp/kbeval ingest ./docs ./README.md
	.venv/bin/kb --data-dir /tmp/kbeval eval run eval/golden-paraphrase.yaml \
		--sweep full --report ./eval/report

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
