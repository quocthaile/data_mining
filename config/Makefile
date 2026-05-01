.PHONY: help install dev test lint format clean run all

# Mục tiêu mặc định
.DEFAULT_GOAL := help

help:
	@echo "MOOCCubeX EDA Pipeline - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install dependencies"
	@echo "  make dev            - Install development dependencies"
	@echo "  make conda          - Create conda environment"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Run main analysis pipeline"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Check code quality (flake8)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make coverage       - Generate coverage report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove cache and build files"
	@echo "  make clean-all      - Remove all generated files"
	@echo ""
	@echo "Combined:"
	@echo "  make all            - Setup, lint, test, and run"

# Installation targets
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort
	@echo "Development dependencies installed"

conda:
	@echo "Creating conda environment..."
	conda env create -f config/environment.yml
	@echo "Conda environment created"
	@echo "Activate with: conda activate mooccubex"

# Development targets
run:
	@echo "Running MOOCCubeX EDA Pipeline..."
	cd scripts && python eda_user-video.py
	@echo "Pipeline completed"

test:
	@echo "Running tests..."
	pytest -v --tb=short
	@echo "Tests completed"

lint:
	@echo "Checking code quality..."
	flake8 scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 scripts/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	@echo "Lint check completed"

format:
	@echo "Formatting code..."
	black scripts/ --line-length=100
	isort scripts/ --profile black --line-length=100
	@echo "Formatting completed"

coverage:
	@echo "Generating coverage report..."
	pytest --cov=scripts --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Cleanup targets
clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ htmlcov/ .coverage
	@echo "Cleanup completed"

clean-all: clean
	@echo "Removing all generated files..."
	rm -rf venv/ env/ .venv/
	rm -rf reports/*.png reports/*.log reports/*.txt
	@echo "All generated files removed"

# Combined targets
all: clean install lint test run
	@echo "All tasks completed successfully!"

.PHONY: help install dev conda run test lint format coverage clean clean-all all
