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
│   ├── api.py                # App factory, middleware, lifespan
│   ├── gcp_utils.py          # Manages gcp resources
│   ├── globals.py            # defines app-wide variables
│   ├── health_check.py       # health check function
│   ├── inference.py          # completions generation, prompt formatting
│   ├── metrics.py            # metric class
│   ├── observability.py      # Collects gpu information
│   ├── trtllm_utils.py       # backend specific utils: trtllm
│   └── vllm_utils.py         # backend specific utils: vllm
├── docker/                   # docker folder containing builds for different backends
│   ├── Dockerfile.vllm
│   ├── Dockerfile.pytorch
│   └── Dockerfile.trtllm
└── requirements/
│   ├── pytorch.txt
│   ├── trtllm.txt
│   └── vllm.txt
└── config/
│   ├── .env.yaml                 # !HIDDEN FILE: project information (hidden file)
│   ├── config.yaml               # !HIDDEN FILE: project config information (endpoint ids, storage ids, model repositories...)
│   ├── build-trtllm-model.yaml   # !HIDDEN FILE: trtllm server information
│   ├── nginx.conf                # nginx server config in case needed
│   └── cloudbuild.yaml           # buildfile containing some basic arguments
└── ci/
│   ├── build_triton_entrypoint_dp.sh             # triton server deployment in a data parallel config. better enable nginx server
│   ├── build_triton_entrypoint_tp.sh             # triton server deployment in a tensor parallel config
│   ├── build_trt_engine.sh                       # building tensorrt_llm engine and upload to gcp
│   ├── deploy_inference_endpoint.sh              # main script that deploys the inference endpoint
│   └── entrypoint-nginx                          # used in the triton server docker image in a data parallel setup

```

---

## Getting Started

The repository serves `meta-llama/Llama-3.2-3B-Instruct` on 3 different inference engines on gcp endpoint.\
Each engine has its own docker image and implements the same base interface defined in `src/api.py`.\
All engines running on a fastapi http server via the single entrypoint `serve.py`:
| Engine/Model | Requirements file | Docker file | Best For |
|---|---|---|--|
| `llama-3.2-vllm` | `requirements/vllm.txt` | `docker/Dockerfile.vllm` | High-throughput, continuous batching |
| `llama-3.2-pytorch` | `requirements/pytorch.txt` | `docker/Dockerfile.pytorch` | Flexibility, research, custom models |
| `llama-3.2-trtllm` | `requirements/trtllm.txt` | `docker/Dockerfile.trtllm` | Low-latency, optimized NVIDIA inference |

ps: All models are loading a huggingface [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)\
Access to the model is only allowed after authorization grant. For more information refer to the huggingface documentation.

(!) Building and deploying the model is only possible after setting up the config files and the gcp environment with proper access rights.\
In particular:
- `gcp-bucket`: GCP storage bucket to contain
- `artifact registry`: to host the docker image
- `model repository`: to create the model + docker image instance
- `inference endpoint`: to deploy the model selected from the repository
Each of these resources need to be properly provisioned along with its dependencies. For example, gpu resources need to be assigned to the inference endpoint to host and run inferences.\
Project config details are hidden because containing sensitive resource ids. However, a [dummy config file](config/dummy-config.yaml) is provided for reference.\
Once all access and credentials are properly set, fill up the dummy config and copy it to config.yaml.

--- 

## Entrypoint

Deploying the entrypoint is done through the `ci/deploy_inference_endpoint.zsh` script.\
Before proceeding with the script, ensure gcloud cli is installed and pointing to the project.\
An example run is as follows

```bash
./deploy_inference_endpoint.zsh \
--region europe-west4 \ 
--model llama-3.2-trtllm \
--update 2
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--region` | `europe-west4` | EU region to host the model repository and deploy the inference endpoint |
| `--model` | `llama-3.2-trtllm` | backend name, refer to Getting started section |
| `--update` | `0` | Level of update: `2` rebuilds all the pipeline, `1` rebuilds the model repository and deploys, `0` only deploys the endpoint |

---

## API Routes

The gateway exposes the following simple API.\
Deploying triton server exposes different APIs.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `predict` | Text completions |

---

## Contributing

Reach out to [mmmarouen](https://github.com/mmarouen)

---

## License

[LICENSE](LICENSE)