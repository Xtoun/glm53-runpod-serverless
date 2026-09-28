# GLM-5.3 Flash RunPod Serverless Panel

Web control panel + llama.cpp backend for **huihui-ai/GLM-5.3-Flash-abliterated-GGUF** on RunPod Serverless.

Designed for a single H200 141 GB worker. The panel starts before the model is downloaded, lets you choose the Hugging Face repo/quant and runtime parameters, downloads the selected GGUF to persistent storage, then starts/stops/restarts llama-server.

## Defaults

- Model repo: `huihui-ai/GLM-5.3-Flash-abliterated-GGUF`
- Quant: `UD-IQ1_S`
- Context per parallel slot: `204800`
- Parallel slots: `4`
- Total llama.cpp context: `819200`
- KV cache: `q8_0 / q8_0`
- GPU layers: `999`
- Flash Attention: enabled

With llama.cpp, the server context is shared between parallel slots. The panel therefore multiplies **context per slot × parallel slots** automatically.

## RunPod

Create an endpoint with **Deploy from a GitHub repository** and select this repository.

Recommended:
- GPU: 141 GB H200
- Min workers: 0 for scale-to-zero, or 1 if you need the panel to stay reachable continuously
- Max workers: 1
- Network Volume mounted at `/runpod-volume`
- HTTP/container port: `8000`

Optional environment variables:

```env
PANEL_PORT=8000
MODEL_ROOT=/runpod-volume/models
STATE_DIR=/runpod-volume/glm53-panel
HF_TOKEN=
CLOUDFLARED_ENABLED=true
```

For a gated/private Hugging Face model, store `HF_TOKEN` as a RunPod Secret.

## First start

The container prints lines similar to:

```text
PANEL USER: admin
PANEL PASSWORD: <random>
API KEY: glm_<random>
CLOUDFLARE PANEL: https://xxxxx.trycloudflare.com
OPENAI BASE URL: https://xxxxx.trycloudflare.com/v1
```

Open the panel, authenticate with `admin` and the generated password, configure the model, click **Download**, then **Start model**.

## Cursor

Use:

```text
Base URL: https://xxxxx.trycloudflare.com/v1
API Key:  glm_...
```

The API is OpenAI-compatible and proxies requests to llama-server.

## Serverless note

With Min Workers = 0 the worker can be destroyed after idle time. The Network Volume keeps the downloaded model and saved configuration, but the Cloudflare Quick Tunnel URL and randomly generated credentials change after every cold start.

## llama.cpp build

The Dockerfile builds current `ggml-org/llama.cpp` `master` with CUDA enabled. You can override the build-time `LLAMA_CPP_REF` with another branch or tag if you need to pin a known-good GLM-5.3 build.
