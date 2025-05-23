# Makefile for CMG-Python development

.PHONY: help install install-dev test test-coverage lint format clean build upload docs

help:
	@echo "CMG-Python Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run code linting (flake8)"
	@echo "  format       Format code (black)"
	@echo ""
	@echo "Build:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload       Upload to PyPI (requires credentials)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pip install -e .[visualization]

test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=cmg --cov-report=html --cov-report=term-missing

lint:
	flake8 cmg/ tests/ examples/ --max-line-length=100 --exclude=__pycache__

format:
	black cmg/ tests/ examples/ --line-length=100

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python setup.py sdist bdist_wheel

upload: build
	twine upload dist/*

docs:
	cd docs && make html

# Development workflow
dev-setup: install-dev
	@echo "Development environment set up!"
	@echo "Run 'make test' to verify installation"

# Quick development check
check: lint test
	@echo "All checks passed!"

# Release workflow
release: check build
	@echo "Package ready for release!"
	@echo "Run 'make upload' to publish to PyPI"
