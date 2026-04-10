import sys
import json
import time
import torch
import asyncio
from transformers import TextIteratorStreamer
from threading import Thread
from dataclasses import dataclass
### local imports
from . import globals
from .metrics import UserMetrics

@dataclass
class MockOutput:
    text: str
    token_ids: list

@dataclass
class MockRequestOutput:
    outputs: list # List of MockOutput
    prompt_token_ids: list

_gpu_lock = asyncio.Lock()

async def pytorch_gen(streaming: bool, max_new_tokens: int, prompt):
        
        if streaming:
            async with _gpu_lock:
                inputs = globals.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_tokens = inputs["input_ids"].shape[1]

                streamer = TextIteratorStreamer(globals.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
                
                def _generate(**kwargs):
                    with torch.no_grad():
                        globals.model.generate(**kwargs)

                thread = Thread(target=_generate, kwargs=generation_kwargs)
                thread.start()
            t_count = 0
            full_text = ""
            for new_text in streamer:
                full_text += new_text
                t_count += 1
                yield MockRequestOutput(
                    outputs=[MockOutput(text=full_text, token_ids=[0] * t_count)], 
                    prompt_token_ids=[0] * input_tokens
                )
                await asyncio.sleep(0) 
            thread.join()
        else:
            async with _gpu_lock:
                inputs = globals.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_tokens = inputs["input_ids"].shape[1]

                def _generate_batch():
                    with torch.no_grad():
                        outputs = globals.model.generate(**inputs, max_new_tokens=max_new_tokens)
                    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:] 
                    text = globals.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    return text, len(gen_tokens)

                full_text, t_count = await asyncio.to_thread(_generate_batch)
            yield MockRequestOutput(
                outputs=[MockOutput(text=full_text, token_ids=[0] * t_count)], 
                prompt_token_ids=[0] * input_tokens
            )

async def generate_completions(results_generator, start_time, collect_kpis=True, streaming=False):
    ttft = None
    ttft_collected = False
    input_len_collected = False
    token_count = 0
    result = None
    previous_text_len = 0
    user_metric = UserMetrics()
    async for request_output in results_generator:
        token_count = len(request_output.outputs[0].token_ids)
        if not ttft_collected and token_count >= 1 and collect_kpis:
            ttft = (time.time() - start_time)
            ttft_collected = True
            user_metric.ttft = ttft
            if streaming:
                yield f"data: {json.dumps({'meta': {'ttft': ttft}})}\n\n"

        if not input_len_collected and collect_kpis and request_output.prompt_token_ids is not None:
            user_metric.input_tokens = len(request_output.prompt_token_ids)
            input_len_collected = True
            if streaming:
                yield f"data: {json.dumps({'meta': {'input_tokens': user_metric.input_tokens}})}\n\n"

        result = request_output
        if streaming:
            full_text = result.outputs[0].text
            new_text = full_text[previous_text_len:]
            previous_text_len = len(full_text)
            if new_text:
                yield f"data: {json.dumps({'text': new_text})}\n\n"

    if result is None:
        raise RuntimeError("No output produced by model")
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
        log_entry.update(user_metric.to_dict())
        print(json.dumps(log_entry))
        sys.stdout.flush()

    if not streaming:
        yield {"predictions": [{"output": text} | user_metric.to_dict()]}

SYSTEM_PROMPT = 'You are a helpful and concise AI assistant.'

def format_prompt(tokenizer, input_text):
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
