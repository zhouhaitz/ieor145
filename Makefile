# make check      fast tests + lint (run after every change, under ~30s)
# make check-all  adds slow statistical tests (MNL recovery, report card on full baseline)
.PHONY: check check-all lint test test-all
lint:
	ruff check bakery_sim tests
test:
	pytest -m "not slow"
test-all:
	pytest
check: lint test
check-all: lint test-all
