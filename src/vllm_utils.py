from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
from vllm import AsyncEngineArgs, AsyncLLMEngine

def get_vllm_args(request_body: dict, max_new_tokens: int):
    uuid = random_uuid()
    sampling_params = SamplingParams(
        temperature=request_body.get("parameters", {}).get("temperature", 0.7),
        top_p=request_body.get("parameters", {}).get("top_p", 1.0),
        max_tokens=request_body.get("parameters", {}).get("max_tokens", max_new_tokens),
    )
    return uuid, sampling_params

def get_vllm_engine(model_path, model_max_length):
    engine_args = AsyncEngineArgs(
        model=model_path,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=2,
        #speculative_config = speculative_config,
        tokenizer=model_path,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=model_max_length
    )
    return AsyncLLMEngine.from_engine_args(engine_args)
