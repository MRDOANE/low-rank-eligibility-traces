.PHONY: test test-external verify lint check

test:
	python -m unittest discover -s tests -v

test-external:
	PYTHONPATH=external_validation/src python -m unittest discover -s external_validation/tests -v

verify:
	python scripts/verify_frozen_results.py
	python scripts/verify_external_results.py

lint:
	python -m ruff check src tests scripts external_validation/src external_validation/tests

check: test test-external verify lint
