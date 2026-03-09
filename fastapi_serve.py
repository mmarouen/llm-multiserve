from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from threading import Thread
from contextlib import asynccontextmanager
import yaml
import asyncio
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import os
import sys
import time
import uvicorn
import argparse
from src.inference.utils import HealthCheck, download_gcs_folder, format_prompt

local_mount = "/tmp/model"
model = None
tokenizer = None
model_ready = False
max_new_tokens = 500
MODEL_NAME = 'llama-3.2-fastapi'
#model_version = 5
health_checker = HealthCheck()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model-version", type=int)

args = parser.parse_args()

with open(os.path.join('config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    # download model from GCS
    print("Downloading model from GCS…")
    download_gcs_folder(
        config['models'][MODEL_NAME]['storage']['bucket'],
        config['models'][MODEL_NAME]['storage']['relative-path'],
        local_mount
    )
    print(f"Loaded model files: {os.listdir(local_mount)}")
    # load model on GPU
    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        local_mount, 
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(local_mount)

    health_checker.is_ready = True
    print(f"Fast api model version {args.model_version} ready!")

    yield

    # cleanup (optional)
    print("Shutting down…")
    model = None
    tokenizer = None
    health_checker.is_ready = False

app = FastAPI(lifespan=lifespan)

health_route = os.environ.get("AIP_HEALTH_ROUTE", "/health")
predict_route = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

@app.get(health_route)
async def health():
    return await health_checker()

def generate_text(inputs):
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def stream_completions(inputs, collect_kpis=True):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    def _generate(**kwargs):
        with torch.no_grad():
            model.generate(**kwargs)

    thread = Thread(target=_generate, kwargs=generation_kwargs)
    
    start_time = time.time()
    thread.start()

    ttft = None
    ttft_collected = False
    token_count = 0
    try:
        for new_text in streamer:
            if (not ttft_collected) and collect_kpis:
                ttft_collected = True
                ttft = (time.time() - start_time) * 1000
                # Optional: Send TTFT as a hidden metadata chunk to the client
                yield f"data: {json.dumps({'meta': {'ttft': ttft}})}\n\n"
            
            token_count += 1
            # Yield the actual text chunk for the UI
            yield f"data: {json.dumps({'text': new_text})}\n\n"
    finally:
        thread.join()
        end_time = time.time()

        if collect_kpis and ttft_collected:
            total_latency = (end_time - start_time)
            # Calculate TPS (Tokens Per Second)
            tps = token_count / (total_latency - (ttft/1000)) if total_latency > (ttft/1000) else 0
            
            # Structured log for GCP Log-based Metrics
            log_entry = {
                "severity": "INFO",
                "message": "inference_stats",
                "ttft": ttft,
                "tps": tps,
                "tokens": token_count
            }
            print(json.dumps(log_entry))
            sys.stdout.flush()

@app.post(predict_route)
async def predict(request: Request):
    if not health_checker.is_ready:
        print('Model not ready')
        return JSONResponse(status_code=503, content={"error": "model loading"})

    body = await request.json()
    instances = body.get("instances", [])
    accept_header = request.headers.get("accept", "")
    is_stream_request = "text/event-stream" in accept_header
    is_stream_payload = body.get("stream", False) # Fallback trigger
    text_input = instances[0]["text"] if instances else body.get("text")
    print(f'Recovered input text {text_input}')

    start = time.time()
    formatted_prompt = format_prompt(tokenizer=tokenizer, input_text=text_input)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    if is_stream_request or is_stream_payload:
            print(f'Received incoming streaming request for text: {text_input}')
            collect_kpis = body.get("collect_kpis", False)
            
            return StreamingResponse(
                stream_completions(inputs, collect_kpis=collect_kpis), 
                media_type="text/event-stream"
            )

    print('Received incoming batch request')
    result = await asyncio.to_thread(generate_text, inputs)
    latency = time.time() - start

    # Vertex AI expects the response in a "predictions" key
    return {
        "predictions": [{
            "output": result, 
            "latency": f"{latency:.4f}s"
        }]
    }

if __name__ == "__main__":
    # Vertex AI injects AIP_HTTP_PORT (usually 8080)
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)