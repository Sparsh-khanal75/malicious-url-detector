# syntax=docker/dockerfile:1
# ---- 1. build: install the package and its dependencies into /install ----------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install .

# ---- 2. train: download the dataset (checksum-verified) and train the model ----------
FROM python:3.12-slim AS trainer
COPY --from=builder /install /usr/local
WORKDIR /work
# Empty = full dataset (~6 min). CI passes TRAIN_SAMPLE=40000 for a fast build.
ARG TRAIN_SAMPLE=
RUN python -m urlguard.train --skip-random-split \
      ${TRAIN_SAMPLE:+--sample $TRAIN_SAMPLE} \
      --out /work/models/urlguard.joblib --reports /work/reports

# ---- 3. runtime: small image, non-root user, no dataset, no build tools --------------
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="urlguard" \
      org.opencontainers.image.description="Malicious URL detection API"
RUN useradd --create-home --uid 10001 app
COPY --from=builder /install /usr/local
COPY --from=trainer --chown=app:app /work/models /app/models
COPY --from=trainer --chown=app:app /work/reports /app/reports
WORKDIR /app
USER app
ENV MODEL_PATH=/app/models/urlguard.joblib PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "urlguard.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
