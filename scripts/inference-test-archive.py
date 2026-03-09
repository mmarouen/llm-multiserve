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
    # Short Input, Long Output (Generation Stress)
    "Write a 500-word technical manual on the calibration process of LiDAR sensors in autonomous vehicles.",
    "Explain the evolution of ISO 26262 functional safety standards from 2011 to today in extreme detail.",
    
    # Long Input (Prefill Stress)
    "Summarize the following sensor logs and identify anomalies: " + ("ERR_VOLTAGE_DROP_01 " * 100),
    "Review this hypothetical Python code for a cruise control system and find 5 security vulnerabilities: \n" + 
    "def control_speed(current, target):\n    if current < target: return current + 5\n" * 50,
    
    # Complex Reasoning (Logic Stress)
    "If a vehicle is traveling at 60km/h and the camera detects an obstacle at 40 meters, calculate the required deceleration if the latency of the braking system is 200ms.",
    "Compare and contrast the edge-case handling of Tesla's FSD vs. Waymo's approach in urban environments.",
    
    # Random/Diversified (Breaking the cache)
    f"Generate a unique VIN number and a corresponding maintenance schedule for a truck in {random.choice(['Norway', 'Brazil', 'Japan'])}.",
]

# Stress Test Stages: (Number of Concurrent Workers, Duration in Seconds)
STAGES = [
    (1, 10),   # Baseline
    (2, 30),   # Low Load
    (5, 30),  # Medium Load
    (10, 60),  # Heavy Stress
    (20, 120)
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

def worker_loop(stop_time):
    """Continuously sends requests until the stage time is up."""
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
            res = run_pipeline_rest(data, endpoint_path=endpoint_path, resource_path=resource_path, stream=STREAM, session=session)
            for _ in res:
                pass
        except Exception as e:
            print(f"Request failed: {e}")

def run_stress_test():
    print(f"Starting Stress Test for Endpoint: {endpoint['id']}")

    for concurrency, duration in STAGES:
        print(f"\n--- STAGE: {concurrency} Concurrent Users for {duration}s ---")
        if TRACE:
            run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=False)
        stop_time = time.time() + duration
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Start 'concurrency' number of threads
            for _ in range(concurrency):
                executor.submit(worker_loop, stop_time)
            # Wait for the duration of the stage
            while time.time() < stop_time:
                time.sleep(1)
        if TRACE:
            run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=True)
    print("\nStress test complete. Check your GCP Dashboard for the results.")

if __name__ == "__main__":
    run_stress_test()
