# syntax=docker/dockerfile:1
# CardioRisk prediction API. Build after training (needs models/model.joblib):
#   docker build -t cardiorisk-api:1.0.0 .
#   docker run --rm -p 8000:8000 cardiorisk-api:1.0.0

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY cardiorisk/ cardiorisk/
COPY app/ app/
COPY models/model.joblib models/model_card.json models/

# Unprivileged runtime user (matches runAsUser in the Kubernetes manifest).
RUN useradd --system --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# One worker per container: Prometheus metrics live in process memory, so scale
# with replicas rather than workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
