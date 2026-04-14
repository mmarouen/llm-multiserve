import os
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SchedulerConfig, CapacitySchedulerPolicy
 
 
def get_trtllm_engine(
    model_path: str,
    # build flags
    max_input_len: int,
    max_seq_len: int,
    max_batch_size: int,
    max_num_tokens: int,
 
    # Parallelism
    tensor_parallel_size: int = 2,
    enable_dp: bool = False,

    # tensorrt_llm/config.pbtxt fields
    gpu_memory_fraction: float = 0.9,
    enable_kv_cache_reuse: bool = True,
    enable_chunked_prefill: bool = True,
    max_queue_delay_us: int = 0,
    max_queue_size: int = 0,
) -> LLM:
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
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        #plugin_config=plugin_config,
    )
    # KV cache config 
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=enable_kv_cache_reuse,
        free_gpu_memory_fraction=gpu_memory_fraction,
    )
 
    # Scheduler config 
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        max_queue_delay_microseconds=max_queue_delay_us, 
        max_queue_size=max_queue_size if max_queue_size > 0 else None,
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
    async for output in engine.generate_async(prompt, sampling_params, streaming=True):
        yield output
