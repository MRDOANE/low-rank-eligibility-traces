.PHONY: test verify lint check

test:
	python -m unittest discover -s tests -v

verify:
	python scripts/verify_frozen_results.py

lint:
	python -m ruff check src tests scripts/verify_frozen_results.py

check: test verify lint
