# Slim runtime image. Base dependencies only — the optional provider extras
# (hosted embeddings, local cross-encoders, YouTube) are opt-in at build time via
# EXTRAS, because torch alone would multiply the image size.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KB_DATA_DIR=/data

WORKDIR /app

# Copy metadata first so the dependency layer caches independently of source.
COPY pyproject.toml README.md ./
COPY backend ./backend

ARG EXTRAS=""
RUN pip install --no-cache-dir ".${EXTRAS}"

# Run unprivileged, and give the data volume to that user.
RUN useradd --create-home --uid 10001 kb && mkdir -p /data && chown -R kb:kb /data
USER kb
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["kb", "serve", "--host", "0.0.0.0", "--port", "8000"]
