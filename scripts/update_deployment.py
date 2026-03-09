import os
import yaml
import argparse
import subprocess
from datetime import datetime

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model-version", type=int)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')
parser.add_argument('--description', type=str)

args = parser.parse_args()

MIN_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1
N_GPUS = 2
DISPLAY_NAME = args.description
GPUS='l4x2'
MODEL_VERSION = args.model_version
MODEL = args.model_name
SPOT = True

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    endpt_id = config['models'][MODEL]['endpoint']['id']
    registry_id = config['models'][MODEL]['registry-id']
    gpus = config['compute'][GPUS]

    now = datetime.now().strftime("%m%d%H%M")
    prefix = 10 if MODEL == 'llama-3.2-fastapi' else 20
    DEPLOYMENT_ID = int(f'{prefix}{now}')

    shell_string = f'gcloud ai endpoints deploy-model {endpt_id}\
  --verbosity=debug \
  --model={registry_id}@{MODEL_VERSION} \
  --region={config["project"]["region"]} \
  --display-name="{DISPLAY_NAME}" \
  --machine-type={gpus["machine_type"]} \
  --accelerator=type={gpus["accelerator_type"]},count={N_GPUS} \
  --deployed-model-id={DEPLOYMENT_ID} \
  --min-replica-count={MIN_REPLICA_COUNT} \
  --max-replica-count={MAX_REPLICA_COUNT} '
    if SPOT:
        shell_string += '--spot'
    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1) # Stop the whole pipeline
