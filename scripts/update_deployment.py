import os
import time
import yaml
import argparse
import subprocess
from datetime import datetime
from src.gcp_utils import get_latest_model_version

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='llama-3.2-vllm')
parser.add_argument('--description', type=str)
parser.add_argument('--region', type=str, default='europe-west3')
parser.add_argument('--spot', type=str2bool, default=False)

args = parser.parse_args()

MIN_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1
N_GPUS = 2
GPUS='l4x2'

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    endpt_id = config['models'][args.model_name]['endpoint']['id'][args.region]
    registry_id = config['models'][args.model_name]['registry-id'][args.region]
    gpus = config['compute'][GPUS]
    model_version = get_latest_model_version(project['id'], args.region, registry_id)
    now = datetime.now().strftime("%m%d%H%M")
    prefix = 10 if args.model_name == 'llama-3.2-pytorch' else 20
    DEPLOYMENT_ID = int(f'{prefix}{now}')
    start_time = time.time()
    shell_string = f'gcloud ai endpoints deploy-model {endpt_id}\
  --verbosity=debug \
  --model={registry_id}@{model_version} \
  --region={args.region} \
  --display-name="{args.description}" \
  --machine-type={gpus["machine_type"]} \
  --accelerator=type={gpus["accelerator_type"]},count={N_GPUS} \
  --deployed-model-id={DEPLOYMENT_ID} \
  --min-replica-count={MIN_REPLICA_COUNT} \
  --max-replica-count={MAX_REPLICA_COUNT} \
  --service-account={project["service-account"]} '
    if args.spot:
        shell_string += '--spot'
    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1) # Stop the whole pipeline
    time_diff = time.time() - start_time
    print(f'Deployment successful after {time_diff:.4f}')