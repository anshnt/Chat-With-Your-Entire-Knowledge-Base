.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install install-ui test test-cov lint fmt typecheck check diagrams links docs serve ui clean demo eval map

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
	.venv/bin/ruff check backend tests scripts

fmt: ## Auto-fix lint findings and format
	.venv/bin/ruff check backend tests scripts --fix
	.venv/bin/ruff format backend tests scripts

typecheck: ## Type-check with mypy
	.venv/bin/mypy

diagrams: ## Check the README's mermaid diagrams parse
	$(PY) scripts/check_mermaid.py

links: ## Check the documentation's internal links resolve
	$(PY) scripts/check_links.py

docs: diagrams links ## Run every documentation check

check: lint typecheck docs test ## Lint, type-check, check the docs, and test

serve: ## Run the API with auto-reload
	.venv/bin/kb serve --reload

install-ui: ## Install the frontend dependencies
	cd frontend && npm install

ui: ## Run the frontend dev server (needs `make serve` in another shell)
	cd frontend && npm run dev

demo: ## Ingest this repo's own docs and run a sample search
	.venv/bin/kb ingest ./docs ./README.md
	.venv/bin/kb stats
	.venv/bin/kb search "how does reciprocal rank fusion combine rankings?"

eval: ## Ingest this repo's docs and run the paraphrase retrieval sweep
	.venv/bin/kb --data-dir /tmp/kbeval ingest ./docs ./README.md
	.venv/bin/kb --data-dir /tmp/kbeval eval run eval/golden-paraphrase.yaml \
		--sweep full --report ./eval/report

map: ## Render the corpus map for this repo's own docs
	.venv/bin/kb --data-dir /tmp/kbmap ingest ./docs ./README.md
	.venv/bin/kb --data-dir /tmp/kbmap map -o docs/assets/corpus-map.svg

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
