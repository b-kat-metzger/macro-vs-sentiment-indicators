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

clean:
	rm -rf data/raw/*
	rm -rf data/clean/*