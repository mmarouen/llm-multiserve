# 🚀 llm-multiserve

A FastAPI-based LLM serving layer with pluggable inference backends, custom routing, and GCP deployment support.

---

## Overview

`llm-multiserve` is a unified serving interface for large language models. It abstracts over multiple inference engines behind a single FastAPI application, letting you swap backends without changing your API surface.

**Key features:**

- 🔌 **Pluggable backends** — switch between vLLM, PyTorch, and TensorRT-LLM via config
- 🛣️ **Custom routing** — define per-model or per-use-case route logic
- 🐳 **Engine-specific Docker images** — each backend ships as its own container
- ☁️ **GCP-native** — designed for deployment on GCP inference endpoints
- 🎯 **Single entrypoint** — everything starts from `serve.py`

---

## Architecture

```
llm-multiserve/
├── serve.py                  # Single entrypoint — starts the FastAPI app
├── src/
│   ├── main.py               # App factory, middleware, lifespan
│   ├── routers/              # Custom route definitions
│   │   ├── completions.py
│   │   ├── chat.py
│   │   └── health.py
│   ├── engines/              # Engine abstractions
│   │   ├── base.py           # Abstract engine interface
│   │   ├── vllm_engine.py
│   │   ├── pytorch_engine.py
│   │   └── trtllm_engine.py
│   └── config.py             # Engine selection + runtime config
├── docker/
│   ├── Dockerfile.vllm
│   ├── Dockerfile.pytorch
│   └── Dockerfile.trtllm
└── deploy/
    └── gcp/                  # GCP endpoint configs
```

---

## Inference Engines

Each engine has its own Docker image and implements the same base interface defined in `engines/base.py`. The active engine is selected at startup via the `INFERENCE_ENGINE` environment variable.

| Engine | Image | Best For |
|---|---|---|
| `vllm` | `Dockerfile.vllm` | High-throughput, continuous batching |
| `pytorch` | `Dockerfile.pytorch` | Flexibility, research, custom models |
| `trtllm` | `Dockerfile.trtllm` | Low-latency, optimized NVIDIA inference |

Set the engine at runtime:

```bash
INFERENCE_ENGINE=vllm python serve.py
INFERENCE_ENGINE=trtllm python serve.py
```

---

## Entrypoint

All serving starts from `serve.py`:

```bash
python serve.py \
  --engine vllm \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 \
  --port 8080
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--engine` | `vllm` | Inference backend to use |
| `--model` | — | Model name or path |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8080` | Bind port |
| `--workers` | `1` | Uvicorn workers |

---

## API Routes

The gateway exposes OpenAI-compatible endpoints by default, plus custom routes.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/readiness` | Readiness probe (engine loaded) |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/chat/completions` | Chat completions |
| `GET` | `/v1/models` | List available models |

Custom routes can be added under `app/routers/`.

---

## Docker

Each engine has its own image. Build the one you need:

```bash
# vLLM
docker build -f docker/Dockerfile.vllm -t inference-gateway:vllm .

# PyTorch
docker build -f docker/Dockerfile.pytorch -t inference-gateway:pytorch .

# TensorRT-LLM
docker build -f docker/Dockerfile.trtllm -t inference-gateway:trtllm .
```

Run a container:

```bash
docker run --gpus all -p 8080:8080 \
  -e INFERENCE_ENGINE=vllm \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
  inference-gateway:vllm
```

---

## GCP Deployment

The service is designed for deployment on **GCP inference endpoints** (Vertex AI or Cloud Run GPU).

Images are pushed to Artifact Registry and referenced in the endpoint config:

```bash
# Authenticate
gcloud auth configure-docker <region>-docker.pkg.dev

# Tag and push
docker tag inference-gateway:vllm \
  <region>-docker.pkg.dev/<project>/inference-gateway/vllm:<tag>

docker push <region>-docker.pkg.dev/<project>/inference-gateway/vllm:<tag>
```

GCP endpoint configs live in `deploy/gcp/`. Environment variables for model and engine are injected via the endpoint spec.

---

## Getting Started

> _TODO: Add setup instructions, virtualenv/conda setup, and a quick local test._

---

## Configuration

> _TODO: Document full config schema (`config.py`), env vars, and model loading options._

---

## Logging & Observability

> _TODO: Add logging setup, Cloud Logging integration, and metrics._

---

## Contributing

> _TODO: Add contribution guidelines._

---

## License

> _TODO: Add license._