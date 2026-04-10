import os
import yaml
import argparse
from google.cloud import aiplatform

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')
parser.add_argument('--description', type=str)
parser.add_argument('--region', type=str, default='europe-west3')

args = parser.parse_args()
PROFILE_RUN = False
HEALTH_ROUTE = "/health"
FIRST_TIME = False
if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    model_config = config['models'][args.model_name]
    model_bucket = model_config['storage']['bucket']
    model_path = model_config['storage']['relative-path']
    
    image_repo_name = model_config['image']['repository-name']
    image_name = model_config['image']['image-name']
    inference_config = model_config['inference']
    sequence_length = model_config["inference"]["max_model_length"]
    max_new_tokens = model_config["inference"]["max_new_tokens"]
    inference_batch_size = model_config["inference"]["inference_batch_size"]
    tp = model_config["inference"]["tensor_parallel"]
    max_num_tokens = int(16 * (sequence_length - max_new_tokens))
    max_num_tokens = int(24 * (sequence_length - max_new_tokens))
    tp = 1
    enable_dp = False
    project_id = project['id']
    triton_image_region = 'europe'
    fastapi_image_region = 'europe-west3'
    parent_model = None
    if not FIRST_TIME:
        model_registry_id = model_config['registry-id'][args.region]
        parent_model = f"projects/{project_id}/locations/{args.region}/models/{model_registry_id}"
    aiplatform.init(project=project_id, location=args.region)
    invoke_enabled_model = aiplatform.Model.upload(
        display_name=f"{args.model_name}-{image_name}",
        artifact_uri=f'gs://{model_bucket}/{model_path}/',
        serving_container_image_uri=f"{fastapi_image_region}-docker.pkg.dev/{project_id}/{image_repo_name}/{image_name}:latest",
        serving_container_invoke_route_prefix="/*",
        version_description=args.description,
        serving_container_ports=[8080],
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_shared_memory_size_mb=4_087,
        serving_container_environment_variables={
            "PROFILE_RUN": f"{PROFILE_RUN}",
            "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": "1",
            "VLLM_TORCH_PROFILER_WITH_FLOPS": "1",
            "VLLM_TORCH_PROFILER_DIR": "/tmp/run_profile",
            "MODEL_NAME": args.model_name,
            "HF_BUCKET": f"gs://{model_bucket}/{model_config['storage']['hf-relative-path']}",
            "MAX_MODEL_LENGTH": sequence_length,
            "MAX_NEW_TOKENS": max_new_tokens,
            "MAX_INPUT_TOKENS": inference_config['max_model_length'] - inference_config['max_new_tokens'],
            "MAX_NUM_TOKENS": max_num_tokens,
            "TRITON_MAX_BATCH_SIZE": inference_batch_size,
            "TENSOR_PARALLEL": tp,
            "ENABLE_DP": enable_dp,
            "INSTANCE_COUNT": 2,
            "MAX_QUEUE_DELAY_MS": 0,
            "MAX_QUEUE_SIZE": 128,
            "DECOUPLED_MODE": False,
            "TRITONSERVER_LOG_VERBOSE": "1",
            "SA_EMAIL": project["service-account"],
            "MODEL_ID": model_config['id'],
            "NUM_GPUS": inference_config['num_gous']
        },
        serving_container_args=[],
        sync=True,
        parent_model=parent_model
    )