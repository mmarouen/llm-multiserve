import os
import yaml
import time
import argparse
import subprocess
from src.gcp_utils import get_latest_model_version

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='llama-3.2-trtllm')
parser.add_argument('--region', type=str, default='europe-west4')

FIRST_TIME = False
args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    if args.model_name == 'trtllm-build':
        image_config = config[args.model_name]['image']
        model_version = 1
    else:
        image_config = config['models'][args.model_name]['image']
        registry_id = config['models'][args.model_name]['registry-id'][args.region]
        if FIRST_TIME:
            model_version = 1
        else:
            #model_version = get_latest_model_version(project['id'], args.region, registry_id)
            model_version = 1 # new version

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = f'europe-west3-docker.pkg.dev/{project["id"]}/{image_config["repository-name"]}/{image_config["image-name"]}:latest'
    config_file = os.path.join(project_root, 'config', 'cloudbuild.yaml')

    start_time = time.time()
    shell_string = f'gcloud builds submit {project_root} \
--substitutions=_M_VERSION={model_version},_TAG={tag},\
_DOCKERFILE={image_config["dockerfile"]},_PROJECT_ID={project["id"]} \
--config {config_file}'

    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
    time_diff = time.time() - start_time
    print(f'Repository update successful after {time_diff:.4f}')