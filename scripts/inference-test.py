import requests
import google.auth
import google.auth.transport.requests
from concurrent.futures import ThreadPoolExecutor
import time
import os
import argparse
import yaml
import uuid
import random
import numpy as np
from src.inference import run_pipeline_rest, run_tracing
from src.commons import get_endpoint_paths

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')
args = parser.parse_args()

with open(os.path.join('config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)
PROJECT_ID = config['project']['id']
PROJECT_NR=config['project']['number']
REGION = config['project']['region']
MODEL = args.model_name
endpoint = config['models'][MODEL]['endpoint']
STREAM = False
TRACE = True

resource_path, endpoint_path = get_endpoint_paths(endpoint, PROJECT_NR, REGION, PROJECT_ID)

PROMPT_BANK = [
# 1 — Technical documentation summarization
(
"Below is internal documentation about automotive LiDAR calibration procedures. "
"Read the documentation and produce a concise 200-300 word summary highlighting "
"the calibration steps, safety checks, and potential failure points.\n\n"
+ ("LiDAR calibration requires aligning the sensor with the vehicle coordinate "
   "system and verifying timing synchronization between perception modules. "
   "Calibration logs indicate voltage stability, signal noise thresholds, and "
   "point-cloud density measurements. ") * 30
),

# 2 — RAG-style question answering
(
"You are an automotive diagnostics assistant. Using the context below, answer "
"the question at the end in less than 250 words.\n\n"
"Context:\n"
+ ("Vehicle telemetry logs show repeated voltage dips in the ECU power rail. "
   "These events occur during high CPU load scenarios when the perception stack "
   "processes camera and radar frames simultaneously. ") * 40
+ "\n\nQuestion:\nWhat is the most likely cause of the voltage instability and how could it be mitigated?"
),

# 3 — Log analysis
(
"Analyze the following embedded system logs and explain the most likely system "
"failure cause. Provide a short technical explanation and recommend two fixes.\n\n"
+ ("[WARN] ECU_TEMP_SPIKE detected in module ADAS_CTRL. "
   "[INFO] Fan controller adjusting RPM. "
   "[ERROR] Voltage regulator output unstable. "
   "[DEBUG] Sensor fusion latency increased. ") * 35
),

# 4 — Policy / compliance analysis
(
"Read the compliance document below and identify the three most critical safety "
"requirements for ISO 26262 ASIL-D certification. Provide a concise explanation.\n\n"
+ ("ISO 26262 defines functional safety requirements for automotive electronic "
   "systems. The standard specifies hazard analysis procedures, risk "
   "classification, and verification steps across development stages. ") * 40
),

# 5 — Incident report summarization
(
"You are reviewing a vehicle incident report. Summarize the root cause and "
"provide two recommendations for preventing future incidents.\n\n"
+ ("Incident timestamp 14:03:22. Vehicle detected pedestrian crossing but "
   "braking response was delayed by 220 milliseconds due to sensor fusion "
   "queue congestion. The perception module attempted fallback processing "
   "using radar data but object classification confidence dropped. ") * 30
),

]

# Stress Test Stages: (Number of Concurrent Workers, Duration in Seconds)
STAGES = [
    (1, 30),   # Baseline
    (2, 30),   # Low Load
    (5, 60),  # Medium Load
    (10, 60),  # Heavy Stress
    (20, 120),
    (40, 240),
]

credentials, _ = google.auth.default()
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)

# 2. Use a Session for connection pooling
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
})

metrics = []

def worker_loop(stop_time):

    while time.time() < stop_time:
        base_prompt = random.choice(PROMPT_BANK)
        # 2. Add a unique ID to ensure no cache hits
        unique_prompt = f"{base_prompt} [Request ID: {uuid.uuid4()}]"
        data = {
            "instances": [{"text": unique_prompt}],
            "stream": STREAM,
            "collect_kpis": True
        }
        try:
            response_iter = run_pipeline_rest(data, endpoint_path=endpoint_path, resource_path=resource_path, stream=STREAM, session=session)
            result = next(response_iter)

            pred = result["predictions"][0]
            tps = pred.get("tps", 0)
            metrics.append({
                "latency": pred.get("latency", 0),
                "ttft": pred.get("ttft", 0),
                "output_tokens": pred.get("output_tokens", 0),
                "input_tokens": pred.get("input_tokens", 0),
                "tps": tps
            })
        except Exception as e:
            print(f"Request failed: {e}")

def summarize(stage_duration):

    if not metrics:
        print('No metrics found')
        return

    latencies = [m["latency"] for m in metrics]
    ttfts = [m["ttft"] for m in metrics]
    output_tokens = [m["output_tokens"] for m in metrics]
    input_tokens = [m["input_tokens"] for m in metrics]

    total_tokens = sum(output_tokens)

    system_tps = total_tokens / stage_duration

    print("requests:", len(metrics))
    print("tokens generated:", total_tokens)

    print("system TPS:", round(system_tps, 2))

    print("TTFT p50:", round(np.median(ttfts), 3))
    print("TTFT p95:", round(sorted(ttfts)[int(len(ttfts) * 0.95)], 3))

    print("latency p50:", round(np.median(latencies), 3))
    print("latency p95:", round(sorted(latencies)[int(len(latencies) * 0.95)], 3))

    print("output tokens p50:", round(np.median(output_tokens), 3))
    print("output tokens p95:", round(sorted(output_tokens)[int(len(output_tokens) * 0.95)], 3))

    print("input tokens p50:", round(np.median(input_tokens), 3))
    print("input tokens p95:", round(sorted(input_tokens)[int(len(input_tokens) * 0.95)], 3))


def run_stress_test():
    print(f"Starting Stress Test for Endpoint: {endpoint['id']}")

    for concurrency, duration in STAGES:
        metrics.clear()
        print(f"\n--- Stage: {concurrency} users for {duration}s ---")
        if TRACE:
            run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=False)
        stop_time = time.time() + duration
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Start 'concurrency' number of threads
            for _ in range(concurrency):
                executor.submit(
                    worker_loop,
                    stop_time
                )
            # Wait for the duration of the stage
            while time.time() < stop_time:
                time.sleep(1)
        summarize(duration)
        if TRACE:
            run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=True)
    print("\nStress test complete. Check your GCP Dashboard for the results.")

if __name__ == "__main__":
    run_stress_test()
