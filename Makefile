# QC Cancellations App Makefile

.PHONY: help setup dev test clean install-deps test-deps create-test-data

help:
	@echo "QC Cancellations App - Available Commands:"
	@echo "  setup          - Set up virtual environment and install dependencies"
	@echo "  dev            - Run the Streamlit app in development mode"
	@echo "  test           - Test installation and dependencies"
	@echo "  clean          - Clean up temporary files"
	@echo "  install-deps   - Install Python dependencies"
	@echo "  test-deps      - Test that all dependencies are working"
	@echo "  create-test-data - Create sample test ZIP file with bucket screenshot"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete! Activate with: source .venv/bin/activate"

dev:
	streamlit run app.py

test: test-deps
	@echo "Running installation test..."
	python test_installation.py

clean:
	rm -rf .venv
	rm -f test_cancellation_packet.zip
	rm -rf __pycache__
	rm -rf .streamlit

install-deps:
	pip install -r requirements.txt

test-deps:
	python test_installation.py

create-test-data:
	python3 create_simple_test_data.py

# For macOS users - install Tesseract
install-tesseract:
	brew install tesseract

# Full setup including Tesseract
setup-full: install-tesseract setup

# Run with test data
run-test: create-test-data dev
