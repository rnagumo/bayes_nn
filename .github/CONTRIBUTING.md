
# Contribution

## Python version

[`pyenv`](https://github.com/pyenv/pyenv) will enable you to manage the multiple versions of Python.

## Dependency management

[`Poetry`](https://python-poetry.org/) will help you with Python packaging and dependency management.

## Testing

[`pytest`](https://docs.pytest.org/en/stable/) will automatically execute unit tests.

```bash
poetry run pytest
```

## Formatting

Formatting will be executed by [`mypy`](https://github.com/python/mypy) for statistic type check, [`flake8`](https://flake8.pycqa.org/en/latest/) for PEP8 rule check, [`isort`](https://pycqa.github.io/isort/) for sorting imports, and [`black`](https://black.readthedocs.io/en/stable/) for formatting.

```bash
poetry run mypy .
poetry run flake8 .
poetry run isort .
poetry run black .
```
