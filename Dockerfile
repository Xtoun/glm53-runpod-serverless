FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip \
    && pip3 install --break-system-packages --no-cache-dir "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /app/llama-server /usr/local/bin/llama-server

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_REPO=huihui-ai/GLM-5.3-Flash-abliterated-GGUF \
    MODEL_QUANT=UD-IQ1_S \
    MODEL_ROOT=/runpod-volume/models \
    CONTEXT_PER_SLOT=204800 \
    PARALLEL=4 \
    GPU_LAYERS=999 \
    KV_CACHE_K=q8_0 \
    KV_CACHE_V=q8_0 \
    FLASH_ATTN=on \
    BATCH_SIZE=2048 \
    UBATCH_SIZE=512 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
