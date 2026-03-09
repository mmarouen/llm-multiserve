import json
import os
import yaml
import time
import requests
import argparse
import google.auth
import google.auth.transport.requests
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
credentials, _ = google.auth.default()
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)

headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json",
    "Connection": "close" # Force connection closure
}
# 2. Use a Session for connection pooling
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
})

if __name__ == "__main__":
    data = {
        "instances": [
            {
                "text": "Write a short summary about the extinction of the dinosaurs. Include only confirmed scientific knowledge"
            }
        ],
        "stream": STREAM,
        "collect_kpis": True
    }
    if TRACE:
        run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=False)
        time.sleep(1.)
    if STREAM:
        print("--- Streaming ---")
        for chunk in run_pipeline_rest(data, endpoint_path=endpoint_path, resource_path=resource_path, session=session, stream=True):
            clean_chunk = chunk[len("data: "):].strip()
            if not clean_chunk:
                    continue
            try:
                parsed_chunk = json.loads(clean_chunk)
                # 3. Access dictionary keys safely
                if "meta" in parsed_chunk:
                    print(f"\nTTFT: {parsed_chunk['meta']['ttft']:.2f} ms")

                if "text" in parsed_chunk:
                    print(parsed_chunk["text"], end="", flush=True)
                        
            except json.JSONDecodeError:
                print(f"\nFailed to parse chunk: {clean_chunk}")
    else:
        # Non streaming
        print("--- Unary ---")
        start = time.time()
        result = run_pipeline_rest(data, endpoint_path=endpoint_path, resource_path=resource_path, session=session, stream=False)
        result = next(result)
        client_side_latency = time.time() - start
        print(f"Latency server side {result['predictions'][0]['latency']}\n\
Latency client side {client_side_latency}\n\
Ttft {result['predictions'][0]['ttft']}")
    if TRACE:
        run_tracing(endpoint_path=endpoint_path, resource_path=resource_path, session=session, stop=True)
