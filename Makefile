.PHONY: help test lint ci

REEL_DIR ?= reel

help:
	@printf "%s\n" \
	  "Targets:" \
	  "  make test  - run integration tests" \
	  "  make lint  - run static checks" \
	  "  make ci    - run lint + test (CI entrypoint)"

test:
	$(MAKE) -C "$(REEL_DIR)" test

lint:
	$(MAKE) -C "$(REEL_DIR)" lint

ci: lint test
