import os
import time
import torch
import asyncio
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
## local imports
from . import globals
from .gcp_utils import download_gcs_folder
from .inference import generate_completions, format_prompt
if globals.use_vllm:
    from .vllm_utils import get_vllm_engine, get_vllm_args
elif globals.use_trtllm:
    from .trtllm_utils import get_trtllm_engine, get_trtllm_args, trtllm_generate
else:
    from .inference import pytorch_gen

def create_lifespan(
        serving: str,
        storage_config: dict,
        local_mount: str,
        max_input_len: int,
        enable_dp: bool,
        max_model_length: int,
        max_num_tokens: int,
        tensor_parallel: int,
        max_queue_delay_us: int,
        max_queue_size: int,
        gpu_memory_fraction: float=0.9,
        max_batch_size: int=64
        ):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Downloading model from GCS…")
        download_gcs_folder(
            storage_config['bucket'],
            storage_config['hf-relative-path'],
            local_mount
        )
        print(f"Loaded model files: {os.listdir(local_mount)}")
        globals.tokenizer = AutoTokenizer.from_pretrained(local_mount)

        if serving == 'vllm':
            globals.engine = get_vllm_engine(model_path=local_mount, model_max_length=max_model_length)
        elif serving == 'pytorch':
            globals.model = AutoModelForCausalLM.from_pretrained(
                local_mount, 
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
                device_map="auto"
            )
        elif serving == 'trtllm':
            import tensorrt_llm
            print(f'Deploying trtllm version: {tensorrt_llm.__version__}')
            globals.engine = get_trtllm_engine(
                model_path=local_mount,
                max_input_len=max_input_len,
                max_seq_len=max_model_length,
                max_batch_size=max_batch_size,
                max_num_tokens=max_num_tokens,
                enable_dp=enable_dp,
                tensor_parallel_size=tensor_parallel,
                max_queue_delay_us=max_queue_delay_us,
                max_queue_size=max_queue_size,
                gpu_memory_fraction=gpu_memory_fraction
            )
        print(f"Loaded {serving} LLM engine")
        globals.health_checker.is_ready = True
        yield

        print("Shutting down…")
        globals.tokenizer = None
        globals.health_checker.is_ready = False
        globals.model = None
        globals.engine = None

    return lifespan

def predict_fn(serving: str, max_new_tokens: int):
    async def predict(request: Request):
        if not globals.health_checker.is_ready:
            print('Model not ready')
            return JSONResponse(status_code=503, content={"error": "model loading"})
        start_time = time.time()
        body = await request.json()
        instances = body.get("instances", [])
        streaming_context = "text/event-stream" in request.headers.get("accept", "") or body.get("stream", False)
        collect_kpis = body.get("collect_kpis", False)
        text_input = instances[0]["text"] if instances else body.get("text")
        print(f'Recovered input text {text_input}')

        formatted_prompt = format_prompt(tokenizer=globals.tokenizer, input_text=text_input)
        results_generator = None
        if serving == 'vllm':
            request_id, sampling_params = get_vllm_args(request_body=body, max_new_tokens=max_new_tokens)
            results_generator = globals.engine.generate(formatted_prompt, sampling_params, request_id)
        elif serving == 'pytorch':
            results_generator = pytorch_gen(streaming=streaming_context, max_new_tokens=max_new_tokens, prompt=formatted_prompt)
        elif serving == 'trtllm':
            sampling_params = get_trtllm_args(
                request_body=body, max_new_tokens=max_new_tokens
            )
            # trtllm_generate wraps engine.generate_async and yields
            # RequestOutput objects with the same .outputs[0].text / .finished
            # shape that generate_completions() already knows how to handle
            results_generator = trtllm_generate(
                engine=globals.engine,
                prompt=formatted_prompt,
                sampling_params=sampling_params,
            )

        if streaming_context:
                print(f'Received incoming streaming request for text: {text_input}')
                return StreamingResponse(
                    generate_completions(results_generator, start_time, collect_kpis=collect_kpis, streaming=True),
                    media_type="text/event-stream"
                )

        print('Received incoming batch request')
        final_output = None
        async for chunk in generate_completions(results_generator, start_time, collect_kpis=collect_kpis, streaming=False):
            final_output = chunk
        return JSONResponse(final_output)
    return predict
