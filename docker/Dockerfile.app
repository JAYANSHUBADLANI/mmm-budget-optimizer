# Modelling and dashboard image. Kept separate from the Airflow image because Airflow pins
# a long list of dependencies and I did not want those constraints applied to PyMC.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/project/src \
    MMM_PROJECT_ROOT=/opt/project

# g++ is required. PyTensor compiles the model graph to C, and without a compiler it falls
# back to a Python linker that samples an order of magnitude slower.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY tests/ ./tests/
COPY pyproject.toml ./

RUN mkdir -p data/raw data/interim data/processed data/external models reports/figures reports/tables

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
