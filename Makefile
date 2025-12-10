PYTHON := python3

setup:
	@echo "Initializing virutal environment..."
	$(PYTHON) -m venv msienv
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Virtual environment initialized."

fetch:
	$(PYTHON) src/data/run_pipeline.py

visualize:
	$(PYTHON) src/utils/charting.py

build-model:
	rm results.txt || true
	$(PYTHON) src/models/linear_regression.py >> results.txt
	$(PYTHON) src/models/rand_forest.py >> results.txt
	$(PYTHON) src/models/lstm_model.py >> results.txt
	@echo "Model building complete. Results saved to results.txt. Visuals saved in src/visuals/."

clean:
	rm -rf data/raw/*
	rm -rf data/clean/*
	rm -rf src/visuals/*

all: setup clean fetch build-model visualize