.PHONY: help install run-api run-folder test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make run-api      - Run FastAPI server"
	@echo "  make run-folder   - Run folder processing script"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean cache files"

install:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-folder:
	python main.py --input-folder data/input --output-folder data/output

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
