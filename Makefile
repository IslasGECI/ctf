all: check init mutants

.PHONY: \
        check \
        clean \
        format \
        init \
        linter \
        mutants \
        setup \
        tests \
        tests_python

module = ctf

define lint
	pylint \
        --disable=bad-continuation \
        --disable=missing-class-docstring \
        --disable=missing-function-docstring \
        --disable=missing-module-docstring \
        ${1}
endef

check:
	black --check --line-length 100 ${module}
	black --check --line-length 100 tests
	flake8 --max-line-length 100 ${module}
	flake8 --max-line-length 100 tests

clean:
	rm --force --recursive ${module}.egg-info
	rm --force --recursive ${module}/__pycache__
	rm --force --recursive tests/__pycache__
	rm --force .mutmut-cache
	rm --force tests/example_submission.csv
	rm --force tests/test.csv
	rm --force tests/test_dataset1/XXexample_submission.csvXX
	rm --force tests/test_dataset1/XXtest.csvXX
	rm --force tests/test_dataset1/XXtrain.csvXX
	rm --force tests/test_dataset1/example_submission.csv
	rm --force tests/test_dataset1/test.csv
	rm --force tests/test_dataset1/train.csv
	rm --force tests/train.csv

format:
	black --line-length 100 ${module}
	black --line-length 100 tests

init: setup tests

setup: clean
	pip uninstall --yes ${module}
	pip install .

linter:
	$(call lint, ${module})
	$(call lint, tests)

mutants: setup
	mutmut run \
        --paths-to-mutate ${module} \
        --runner "make tests"

tests: tests_python

tests_python:
	pytest --verbose
