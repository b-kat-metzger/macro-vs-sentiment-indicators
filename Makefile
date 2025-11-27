PYTHON := python3

setup:
	@echo "Initializing virutal environment..."
	$(PYTHON) -m venv msienv
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Virtual environment initialized."

fetch:
	$(PYTHON) src/data/fetch_data.py
	$(PYTHON) src/data/clean_data.py

clean:
	rm -rf data/raw/*