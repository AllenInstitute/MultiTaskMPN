# Tests

All automated tests and manual test/check scripts for this repository belong
in this top-level `test/` directory. Do not add test code beside production
modules or at the repository root.

Run the automated suite from the repository root with:

```bash
python -m unittest discover -s test -p 'test_*.py'
```

`biology_data_check.py` is a manual data-dependent check and is intentionally
not named `test_*.py`, so automated discovery does not execute it.
