# Convenience targets. Everything here is a thin wrapper over a script, so nothing is
# hidden behind the Makefile that cannot be run directly.

.PHONY: help setup data replica test lint phase1 phase2 phase3 smoke dashboard dbt clean

help:
	@echo "setup      install python dependencies"
	@echo "data       download the datasets from Kaggle and GitHub"
	@echo "replica    generate the schema faithful stand in dataset"
	@echo "test       run the test suite"
	@echo "lint       run ruff"
	@echo "smoke      run phases 1 and 3 with minimal sampling, end to end in a few minutes"
	@echo "phase1     data quality, EDA, model fit, backtest"
	@echo "phase2     benchmark, generalisation, causal validation"
	@echo "phase3     CPM translation, optimiser, sensitivity"
	@echo "dashboard  launch the Streamlit scenario dashboard"
	@echo "dbt        run the dbt models and tests"

setup:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/fetch_data.py

replica:
	python scripts/make_replica_dataset.py --force

test:
	PYTHONPATH=src python -m pytest tests -q

lint:
	ruff check src tests scripts app

smoke:
	python scripts/run_phase1.py --smoke --no-backtest
	python scripts/run_phase3.py --no-sensitivity

phase1:
	python scripts/run_phase1.py

phase2:
	python scripts/run_phase2.py

phase3:
	python scripts/run_phase3.py

dashboard:
	streamlit run app/streamlit_app.py

dbt:
	cd dbt && dbt deps --profiles-dir . && dbt run --profiles-dir . && dbt test --profiles-dir .

clean:
	rm -rf reports/figures/*.png reports/tables/*.csv reports/tables/*.json
	rm -rf .pytest_cache .ruff_cache dbt/target dbt/logs
	find . -type d -name __pycache__ -exec rm -rf {} +
