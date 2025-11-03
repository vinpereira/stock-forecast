.PHONY: install test run clean streamlit

install:
	uv sync

test:
	uv run pytest -v

run:
	uv run python main.py

clean:
	rm -rf outputs/*.png outputs/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

streamlit:
	uv run streamlit run app.py