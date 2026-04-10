import os
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SchedulerConfig, CapacitySchedulerPolicy
 
 
def get_trtllm_engine(
    model_path: str,
    # -----------------------------------------------------------------------
    # Shape / capacity  →  trtllm-build flags
    # -----------------------------------------------------------------------
    max_input_len: int,                  # --max_input_len         ${MAX_INPUT_TOKENS}
    max_seq_len: int,                    # --max_seq_len           ${MAX_MODEL_LENGTH}
    max_batch_size: int,                 # --max_batch_size        ${MAX_BATCH_SIZE}
    max_num_tokens: int,                 # --max_num_tokens        ${MAX_NUM_TOKENS}
 
    # -----------------------------------------------------------------------
    # Parallelism
    # -----------------------------------------------------------------------
    tensor_parallel_size: int = 2,       # --workers (tp_size)     2
    enable_dp: bool = False,
    # -----------------------------------------------------------------------
    # Runtime / executor  →  tensorrt_llm/config.pbtxt fields
    # -----------------------------------------------------------------------
    gpu_memory_fraction: float = 0.9,    # kv_cache_free_gpu_mem_fraction: 0.9
    enable_kv_cache_reuse: bool = True,  # enable_kv_cache_reuse:          True
    enable_chunked_prefill: bool = True, # enable_chunked_context:         true
    max_queue_delay_us: int = 0,         # max_queue_delay_microseconds:   ${MAX_QUEUE_DELAY_MS}
    max_queue_size: int = 0,             # max_queue_size:                 ${MAX_QUEUE_SIZE}
) -> LLM:
    # Persist compiled engine across restarts — MPI workers inherit this env var
    os.environ.setdefault("TLLM_LLMAPI_BUILD_CACHE", "1")
    '''
    plugin_config = PluginConfig(
        #gpt_attention_plugin="auto",   # --gpt_attention_plugin auto
        #gemm_plugin="auto",            # --gemm_plugin auto
        remove_input_padding=True,     # --remove_input_padding enable
        paged_kv_cache=True,           # --kv_cache_type paged
        use_paged_context_fmha=True,   # --use_paged_context_fmha enable
        context_fmha=True,             # --context_fmha enable
        multiple_profiles=True,        # --multiple_profiles enable
    )
    '''
    build_config = BuildConfig(
        max_input_len=max_input_len,   # --max_input_len
        max_seq_len=max_seq_len,       # --max_seq_len
        max_batch_size=max_batch_size, # --max_batch_size
        max_num_tokens=max_num_tokens, # --max_num_tokens
        #plugin_config=plugin_config,
    )

    # -----------------------------------------------------------------------
    # KV cache config  →  tensorrt_llm/config.pbtxt KV cache fields
    # -----------------------------------------------------------------------
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=enable_kv_cache_reuse,       # enable_kv_cache_reuse
        free_gpu_memory_fraction=gpu_memory_fraction,   # kv_cache_free_gpu_mem_fraction
    )
 
    # -----------------------------------------------------------------------
    # Scheduler config  →  tensorrt_llm/config.pbtxt batching fields
    # -----------------------------------------------------------------------
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,  # batch_scheduler_policy: max_utilization
        max_queue_delay_microseconds=max_queue_delay_us,                     # max_queue_delay_microseconds
        max_queue_size=max_queue_size if max_queue_size > 0 else None,       # max_queue_size
    )

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_chunked_prefill=enable_chunked_prefill,
        build_config=build_config,
        enable_attention_dp=enable_dp,
        kv_cache_config=kv_cache_config,
        scheduler_config=scheduler_config,
    )
 
 
def get_trtllm_args(request_body: dict, max_new_tokens: int) -> SamplingParams:
    """Mirrors get_vllm_args — no request_id needed, executor assigns its own."""
    params = request_body.get("parameters", {})
    return SamplingParams(
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 1.0),
        max_tokens=params.get("max_tokens", max_new_tokens),
    )
 
 
async def trtllm_generate(engine: LLM, prompt: str, sampling_params: SamplingParams):
    """
    Async generator wrapping engine.generate_async.
 
    Yields RequestOutput objects with the same shape generate_completions()
    expects from vLLM:
        output.outputs[0].text   — full decoded text so far
        output.finished          — True on the last chunk
 
    TRT-LLM's RequestOutput is structurally identical to vLLM's for these
    fields, so generate_completions() requires no changes.
    """
    async for output in engine.generate_async(prompt, sampling_params, streaming=True):
        yield output
