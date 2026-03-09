import os
import yaml
import argparse
import subprocess
from google.cloud import aiplatform

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')
parser.add_argument('--description', type=str)

args = parser.parse_args()
PROFILE_RUN = True
HEALTH_ROUTE = '/health'
if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    model_config = config['models'][args.model_name]
    model_bucket = model_config['storage']['bucket']
    model_path = model_config['storage']['relative-path']
    model_registry_id = model_config['registry-id']
    image_repo_name = model_config['image']['repository-name']
    image_name = model_config['image']['image-name']
    inference_config = model_config['inference']
    project_id = config['project']['id']
    region = config['project']['region']

    aiplatform.init(project=project_id, location=region)
    invoke_enabled_model = aiplatform.Model.upload(
        display_name=f"{args.model_name}-{image_name}",
        serving_container_image_uri=f"{region}-docker.pkg.dev/{project_id}/{image_repo_name}/{image_name}:latest",
        serving_container_invoke_route_prefix="/*",
        version_description=f"{args.description}",
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_environment_variables={
            "PROFILE_RUN": "True",
            "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": "1",
            "VLLM_TORCH_PROFILER_WITH_FLOPS": "1",
            "VLLM_TORCH_PROFILER_DIR": "/tmp/run_profile",
            "AIP_HTTP_PORT": "8080",
            "MAX_MODEL_LENGTH": inference_config['max_model_length'],
            "MAX_NEW_TOKENS": inference_config['max_new_tokens']
        },
        serving_container_args=[],
        sync=True,
        parent_model=f"projects/{project_id}/locations/{region}/models/{model_registry_id}"
    )

    '''
    shell_string = f'gcloud ai models upload \
  --region={region} \
  --display-name="{args.model_name}-{image_name}" \
  --container-image-uri="{region}-docker.pkg.dev/{project_id}/{image_repo_name}/{image_name}:latest" \
  --artifact-uri="gs://{model_bucket}/{model_path}/" \
  --version-description="{args.description}" \
  --container-env-vars="AIP_HTTP_PORT=8080" \
  --parent-model="projects/{project_id}/locations/{region}/models/{model_registry_id}"'
    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
    '''