import os
import yaml
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-version', type=int)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')

args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    image_config = config['models'][args.model_name]['image']
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = f'{project["region"]}-docker.pkg.dev/{project["id"]}/{image_config["repository-name"]}/{image_config["image-name"]}:latest'
    config_file = os.path.join(project_root, 'config', 'cloudbuild.yaml')
    shell_string = f'gcloud builds submit {project_root} \
--substitutions=_M_VERSION={args.model_version},_TAG={tag},\
_DOCKERFILE={image_config["dockerfile"]},_PROJECT_ID={project["id"]} \
--config {config_file}'
    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
