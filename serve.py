from fastapi import FastAPI, Request, BackgroundTasks
import yaml
import os
import asyncio
import uvicorn
import argparse
import src.globals as globals

local_mount = "/tmp/model"
MAX_NEW_TOKENS_ = 500
is_profiling = False
MAX_MODEL_LENGTH_=4096
TORCH_PROFILING = os.environ.get("PROFILE_RUN", "False").lower() == "true"
PROFILER_LOCAL_FOLDER = os.environ.get("VLLM_TORCH_PROFILER_DIR", '/tmp/run_profile')
MAX_MODEL_LENGTH = int(os.environ.get("MAX_MODEL_LENGTH", MAX_MODEL_LENGTH_))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_TOKENS", MAX_MODEL_LENGTH_))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", MAX_NEW_TOKENS_))
MAX_NUM_TOKENS = int(os.environ.get("MAX_NUM_TOKENS", MAX_MODEL_LENGTH_))
MAX_BATCH_SIZE = int(os.environ.get("TRITON_MAX_BATCH_SIZE", 64))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", 2))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 64))
MAX_QUEUE_DELAY_MS = int(os.environ.get("MAX_QUEUE_DELAY_MS", 0))
ENABLE_DP = bool(os.environ.get("ENABLE_DP", False))

MODEL_NAME = os.environ.get("MODEL_NAME", 'llama-3.2-vllm')
PROFILE_OUTPUT_FOLDER = f'{MODEL_NAME.replace(".", "-")}-traces'
SERVING = None
if 'vllm' in MODEL_NAME:
    SERVING = 'vllm'
    globals.use_vllm = True
elif 'trtllm' in MODEL_NAME:
    SERVING = 'trtllm'
    globals.use_trtllm = True
else:
    SERVING = 'pytorch'

### local imports
from src.observability import describe_gpus
from src.gcp_utils import export_profile_gcp
from src.api import create_lifespan, predict_fn

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model-version", type=int)
parser.add_argument("--port", type=int, default=8080)

args = parser.parse_args()

print(f'==== Arguments Overview ====:\n\
-- SERVING {SERVING}\n\
-- MAX_MODEL_LENGTH {MAX_MODEL_LENGTH}\n\
-- MAX_INPUT_LENGTH {MAX_INPUT_LENGTH}\n\
-- MAX_NUM_TOKENS {MAX_NUM_TOKENS}\n\
-- MAX_BATCH_SIZE {MAX_BATCH_SIZE}\n\
-- TENSOR_PARALLEL {TENSOR_PARALLEL}\n\
-- MAX_QUEUE_SIZE {MAX_QUEUE_SIZE}\n\
-- MAX_QUEUE_DELAY_MS {MAX_QUEUE_DELAY_MS}\n\
-- TORCH_PROFILING {TORCH_PROFILING}')

print("==== GPUs Overview ====")
description = describe_gpus()
print(describe_gpus)

with open(os.path.join('config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)
with open(os.path.join('config', '.env.yaml'), 'r') as file:
    project = yaml.safe_load(file)['project']

app = FastAPI(
    lifespan=create_lifespan(
        serving=SERVING,
        storage_config=config['models'][MODEL_NAME]['storage'],
        local_mount=local_mount,
        max_model_length=MAX_MODEL_LENGTH,
        enable_dp=ENABLE_DP,
        max_num_tokens=MAX_NUM_TOKENS,
        max_batch_size=MAX_BATCH_SIZE,
        tensor_parallel=TENSOR_PARALLEL,
        max_input_len=MAX_INPUT_LENGTH,
        max_queue_size=MAX_QUEUE_SIZE,
        max_queue_delay_us=MAX_QUEUE_DELAY_MS
        )
    )

@app.get("/health")
async def health():
    return await globals.health_checker()

if SERVING == 'vllm':
    @app.post("/start_profiling")
    async def start_profiling():
        global is_profiling
        if not TORCH_PROFILING:
            return {"status": "error", "message": "PROFILE_RUN environment variable is not True"}
        if is_profiling:
            return {"status": "ignored", "message": "Profiling is already running"}

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, globals.engine.engine.model_executor.start_profile)

        print("Profiler started")
        is_profiling = True
        return {"status": "success", "message": "Profiling started"}

    @app.post("/stop_profiling")
    async def stop_profiling(background_tasks: BackgroundTasks):
        global is_profiling
        if not is_profiling:
            return {"status": "ignored", "message": "Profiling is not currently running"}

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, globals.engine.engine.model_executor.stop_profile)
        print("Profiler stopped")
        is_profiling = False
        
        background_tasks.add_task(
            export_profile_gcp,
            gs_bucket=config['models'][MODEL_NAME]['storage']['bucket'],
            gs_relative_path=PROFILE_OUTPUT_FOLDER,
            profile_folder=PROFILER_LOCAL_FOLDER
        )
        return {"status": "success", "message": f"Profiling stopped. Traces uploading to {PROFILE_OUTPUT_FOLDER} in background."}

prediction_fnc = predict_fn(serving=SERVING, max_new_tokens=MAX_NEW_TOKENS)
@app.post('/predict')
async def predict(request: Request):
    return await prediction_fnc(request)

if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)