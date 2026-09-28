FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake ninja-build build-essential python3 python3-pip \
    libcurl4-openssl-dev libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# cloudflared for the temporary public admin/API URL.
RUN arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
      amd64) cfarch=amd64 ;; \
      arm64) cfarch=arm64 ;; \
      *) echo "Unsupported arch: $arch" && exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cfarch}" \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

WORKDIR /opt
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp.git && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
      -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/llama.cpp/build --target llama-server -j"$(nproc)" && \
    ln -s /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

COPY app /app
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
