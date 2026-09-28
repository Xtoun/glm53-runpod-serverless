# GLM-5.3 Flash on RunPod Serverless

Minimal RunPod Serverless image for:

`huihui-ai/GLM-5.3-Flash-abliterated-GGUF`

There is **no web panel**. The container automatically downloads the selected GGUF quant to the RunPod Network Volume and starts the OpenAI-compatible `llama-server` API.

## Recommended RunPod configuration

Use **Deploy from a GitHub repository**:

- Branch: `main`
- Dockerfile: `/Dockerfile`
- Mode: **Load balancer**
- Health endpoint: `/health`
- GPU: **H200 141 GB**
- Min workers: `0` for scale-to-zero
- Max workers: `1`
- Network Volume mounted at: `/runpod-volume`

The server listens on port `8000`.

## Defaults

```env
MODEL_REPO=huihui-ai/GLM-5.3-Flash-abliterated-GGUF
MODEL_QUANT=UD-IQ1_S

CONTEXT_PER_SLOT=204800
PARALLEL=4

GPU_LAYERS=999
KV_CACHE_K=q8_0
KV_CACHE_V=q8_0
FLASH_ATTN=on

BATCH_SIZE=2048
UBATCH_SIZE=512
PORT=8000
MODEL_ROOT=/runpod-volume/models
```

With the defaults, llama.cpp receives a total context pool of:

`204800 × 4 = 819200 tokens`

This gives up to four parallel slots, each with a target maximum of about 200K tokens.

## Persistent model storage

The GGUF is downloaded to:

`/runpod-volume/models`

On later cold starts, `hf download` reuses the existing files instead of downloading the full model again.

## Hugging Face token

The default model is public, so an HF token should not normally be necessary.

If you use a private/gated model, add `HF_TOKEN` as a RunPod Secret.

## Optional variables

```env
EXTRA_ARGS=
```

`EXTRA_ARGS` is appended to the `llama-server` command.

## OpenAI-compatible API

RunPod Load Balancer routes requests directly to `llama-server`.

Useful paths:

```text
/health
/v1/models
/v1/chat/completions
/v1/completions
```

Use the RunPod Load Balancer endpoint as the base URL in a client that supports an OpenAI-compatible API.

## Scale-to-zero

With `Min workers = 0`, RunPod can shut the H200 worker down when idle.

When a new request arrives:

1. RunPod starts the worker.
2. The GGUF is read from the Network Volume.
3. llama.cpp loads the model into VRAM.
4. `/health` becomes ready.
5. RunPod routes traffic to the worker.

The model does not need to be downloaded again unless the Network Volume is removed or the selected quant changes.
