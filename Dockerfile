FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# The official llama.cpp CUDA image already contains /app/llama-server.
RUN ln -sf /app/llama-server /usr/local/bin/llama-server

# cloudflared for the temporary public admin/API URL.
RUN arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
      amd64) cfarch=amd64 ;; \
      arm64) cfarch=arm64 ;; \
      *) echo "Unsupported arch: $arch" && exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cfarch}" \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

WORKDIR /panel
COPY requirements.txt /panel/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /panel/requirements.txt

COPY app /app-panel
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PANEL_PORT=8000 \
    MODEL_ROOT=/runpod-volume/models \
    STATE_DIR=/runpod-volume/glm53-panel \
    CLOUDFLARED_ENABLED=true \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
