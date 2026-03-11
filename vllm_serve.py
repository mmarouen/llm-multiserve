from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import yaml
import json
from transformers import AutoTokenizer
import torch
import os
import asyncio
import sys
import time
import uvicorn
import argparse
from datetime import datetime
import vllm
import subprocess
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
try:
    import pynvml
except ImportError:
    pynvml = None
    print("pynvml not installed. Some details (NVLink) will be missing.")
from src.commons import get_endpoint_paths, UserMetrics
from src.inference.utils import HealthCheck, download_gcs_folder, format_prompt, export_profile_gcp

local_mount = "/tmp/model"
tokenizer = None
engine = None
MAX_NEW_TOKENS_ = 500
is_profiling = False
MODEL_NAME = 'llama-3.2-vllm'
MAX_MODEL_LENGTH_=4096
#model_version = 5
health_checker = HealthCheck()
PROFILE_OUTPUT_FOLDER = f'{MODEL_NAME.replace(".", "-")}-traces'
TORCH_PROFILING = os.environ.get("PROFILE_RUN", "False").lower() == "true"
PROFILER_LOCAL_FOLDER = os.environ.get("VLLM_TORCH_PROFILER_DIR", '/tmp/run_profile')
MAX_MODEL_LENGTH = int(os.environ.get("MAX_MODEL_LENGTH", MAX_MODEL_LENGTH_))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", MAX_NEW_TOKENS_))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model-version", type=int)

args = parser.parse_args()

print(f'Model generation parameters:\n\
MAX_MODEL_LENGTH {MAX_MODEL_LENGTH}\n\
MAX_NEW_TOKENS {MAX_NEW_TOKENS}')

print("==== GPUs Overview ====")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}\n")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Multi-Processor Count: {props.multi_processor_count}")
    print(f"  CUDA Device Index: {i}\n")

if pynvml:
    try:
        pynvml.nvmlInit()
        labels = {
            pynvml.NVML_TOPOLOGY_INTERNAL:   "NVLink (same board)",
            pynvml.NVML_TOPOLOGY_SINGLE:     "PIX - same PCIe switch",
            pynvml.NVML_TOPOLOGY_MULTIPLE:   "PXB - multiple PCIe switches",
            pynvml.NVML_TOPOLOGY_HOSTBRIDGE: "PHB - PCIe host bridge",
            pynvml.NVML_TOPOLOGY_CPU:        "CPU - same NUMA/CPU",
            pynvml.NVML_TOPOLOGY_SYSTEM:     "SYS - cross-socket",
        }

        print("NVML topology")
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
                handle_j = pynvml.nvmlDeviceGetHandleByIndex(j)
                level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                print(f"GPU{i} <-> GPU{j}: {labels.get(level, level)}")
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print("NVML error:", e)

with open(os.path.join('config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)
with open(os.path.join('config', '.env.yaml'), 'r') as file:
    project = yaml.safe_load(file)['project']

endpoint_path, resource_path = get_endpoint_paths(
    config['models'][MODEL_NAME]['endpoint'],
    project['number'],
    project['region'],
    project['id']
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, engine

    # download model from GCS
    print("Downloading model from GCS…")
    download_gcs_folder(
        config['models'][MODEL_NAME]['storage']['bucket'],
        config['models'][MODEL_NAME]['storage']['relative-path'],
        local_mount
    )
    print(f"Loaded model files: {os.listdir(local_mount)}")
    tokenizer = AutoTokenizer.from_pretrained(local_mount)

    if TORCH_PROFILING:
        print('Torch profiling enabled')
    speculative_config = {
        "method": "ngram",
        "num_speculative_tokens": 2,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }

    engine_args = AsyncEngineArgs(
            model=local_mount,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            tensor_parallel_size=2,
            #speculative_config = speculative_config,
            tokenizer=local_mount,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LENGTH
        )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"vLLM Engine version {args.model_version} is ready!")
    health_checker.is_ready = True
    yield

    print("Shutting down…")
    tokenizer = None
    health_checker.is_ready = False

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return await health_checker()

@app.post("/start_profiling")
async def start_profiling():
    global is_profiling, engine
    if not TORCH_PROFILING:
        return {"status": "error", "message": "PROFILE_RUN environment variable is not True"}
    if is_profiling:
        return {"status": "ignored", "message": "Profiling is already running"}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.engine.model_executor.start_profile)

    print("Profiler started")

    is_profiling = True
    return {"status": "success", "message": "Profiling started"}

@app.post("/stop_profiling")
async def stop_profiling(background_tasks: BackgroundTasks):
    global is_profiling, engine
    if not is_profiling:
        return {"status": "ignored", "message": "Profiling is not currently running"}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.engine.model_executor.stop_profile)
    print("Profiler stopped")
    is_profiling = False
    
    # Generate a unique folder name for this specific trace export
    run_timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    gcs_export_path = f"{PROFILE_OUTPUT_FOLDER}/trace-{run_timestamp}"
    
    # Upload to GCS as a background task so the API responds immediately
    background_tasks.add_task(
        export_profile_gcp,
        gs_bucket=config['models'][MODEL_NAME]['storage']['bucket'],
        gs_relative_path=PROFILE_OUTPUT_FOLDER,
        profile_folder=PROFILER_LOCAL_FOLDER
    )
    return {"status": "success", "message": f"Profiling stopped. Traces uploading to {PROFILE_OUTPUT_FOLDER} in background."}

async def stream_completions(results_generator, start_time, collect_kpis=True):
    user_metric = UserMetrics()

    ttft = None
    ttft_collected = False
    token_count = 0
    previous_text_len = 0
    input_len_collected = False
    async for request_output in results_generator:
        token_count = len(request_output.outputs[0].token_ids)
        if not ttft_collected and token_count >= 1 and collect_kpis:
            ttft = (time.time() - start_time)
            ttft_collected = True
            user_metric.ttft = ttft
            yield f"data: {json.dumps({'meta': {'ttft': ttft}})}\n\n"

        if not input_len_collected and collect_kpis:
            user_metric.input_tokens = len(request_output.prompt_token_ids)
            input_len_collected = True
            yield f"data: {json.dumps({'meta': {'input_tokens': user_metric.input_tokens}})}\n\n"

        full_text = request_output.outputs[0].text
        new_text = full_text[previous_text_len:]
        previous_text_len = len(full_text)
        if new_text:
            yield f"data: {json.dumps({'text': new_text})}\n\n"
    end_time = time.time()

    if collect_kpis and ttft_collected:
        total_latency = (end_time - start_time)
        # Calculate TPS (Tokens Per Second)
        tps = token_count / (total_latency - ttft) if total_latency > ttft else 0
        user_metric.latency = total_latency
        user_metric.tps = tps
        user_metric.output_tokens = token_count
        # Structured log for GCP Log-based Metrics
        log_entry = {
            "severity": "INFO",
            "message": "inference_stats",
        }
        log_entry.update(user_metric.model_dump())
        print(json.dumps(log_entry))
        sys.stdout.flush()

@app.post('/predict')
async def predict(request: Request):
    if not health_checker.is_ready:
        print('Model not ready')
        return JSONResponse(status_code=503, content={"error": "model loading"})
    body = await request.json()
    instances = body.get("instances", [])
    accept_header = request.headers.get("accept", "")
    is_stream_request = "text/event-stream" in accept_header
    is_stream_payload = body.get("stream", False) # Fallback trigger
    collect_kpis = body.get("collect_kpis", False)
    text_input = instances[0]["text"] if instances else body.get("text")
    print(f'Recovered input text {text_input}')

    formatted_prompt = format_prompt(tokenizer=tokenizer, input_text=text_input)
    request_id = random_uuid()
    start_time = time.time()
    sampling_params = SamplingParams(
            temperature=body.get("parameters", {}).get("temperature", 0.7),
            top_p=body.get("parameters", {}).get("top_p", 1.0),
            max_tokens=body.get("parameters", {}).get("max_tokens", MAX_NEW_TOKENS),
        )
    results_generator = engine.generate(formatted_prompt, sampling_params, request_id)
    if is_stream_request or is_stream_payload:
            print(f'Received incoming streaming request for text: {text_input}')
            return StreamingResponse(
                stream_completions(results_generator, start_time, collect_kpis=collect_kpis), media_type="text/event-stream")

    print('Received incoming batch request')
    ttft = None
    ttft_collected = False
    input_len_collected = False
    token_count = 0
    result = None
    user_metric = UserMetrics()
    async for request_output in results_generator:
        
        token_count = len(request_output.outputs[0].token_ids)
        if not ttft_collected and token_count >= 1 and collect_kpis:
            ttft = (time.time() - start_time)
            ttft_collected = True
            user_metric.ttft = ttft
        if not input_len_collected and collect_kpis:
            user_metric.input_tokens = len(request_output.prompt_token_ids)
            input_len_collected = True
        result = request_output
    text = result.outputs[0].text
    latency = time.time() - start_time
    user_metric.output_tokens = token_count
    user_metric.latency = latency
    if collect_kpis:
        decode_time = latency - ttft if ttft else latency
        tps = token_count / decode_time if decode_time > 0 else 0
        user_metric.tps = tps
        log_entry = {
            "severity": "INFO",
            "message": "inference_stats"
        }
        log_entry.update(user_metric.model_dump())
        print(json.dumps(log_entry))
        sys.stdout.flush()

    # Vertex AI expects the response in a "predictions" key
    return {
        "predictions": [{"output": text} | user_metric.model_dump()]
    }

if __name__ == "__main__":
    # Vertex AI injects AIP_HTTP_PORT (usually 8080)
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)