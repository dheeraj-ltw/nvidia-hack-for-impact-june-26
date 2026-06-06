FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ffmpeg encodes recorded sessions and decodes audio for speaker ID; gcc builds the
# webrtcvad C extension that resemblyzer (speaker embeddings) depends on — no wheel exists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    # Use the OS trust store (not certifi) so any CA added via ca-certificates is honored.
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

COPY pyproject.toml ./
RUN uv pip install --system --no-cache .

COPY . .

# Trust any operator-provided root CAs (mounted into ./certs) on startup, then run.
ENTRYPOINT ["/app/docker-entrypoint.sh"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
