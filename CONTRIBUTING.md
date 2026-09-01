# Contributing

Issues and pull requests that improve clarity, tests, portability, or reproducibility are welcome.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
python scripts/verify_frozen_results.py
python -m ruff check src tests scripts/verify_frozen_results.py
```

Frozen result files must not be replaced in place. New experimental evidence requires a new protocol version, output namespace, changelog entry, and release version. Please keep large checkpoints and result archives out of Git history.
