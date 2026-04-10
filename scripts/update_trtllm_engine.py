import os
import yaml
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--compute', type=str, default='l4x2')
args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    job_config_file_path = os.path.join(project_root, 'config', 'build-trtllm-model.yaml')
    with open(job_config_file_path, 'r') as file:
        job_config_file = yaml.safe_load(file)
    model_config = config['models']['llama-3.2-trtllm']
    image_config = config['trtllm-build']['image']
    hf_bucket = f'gs://{model_config["storage"]["bucket"]}/{model_config["storage"]["hf-relative-path"]}'
    model_bucket = f'gs://{model_config["storage"]["bucket"]}/{model_config["storage"]["relative-path"]}'
    sequence_length = model_config["inference"]["max_model_length"]
    max_new_tokens = model_config["inference"]["max_new_tokens"]
    inference_batch_size = model_config["inference"]["inference_batch_size"]
    tp = model_config["inference"]["tensor_parallel"]

    machine_spec = job_config_file["workerPoolSpecs"][0]['machineSpec']
    machine_spec = {}
    machine_spec['machineType'] = config['compute'][args.compute]['machine_type']
    machine_spec['acceleratorType'] = config['compute'][args.compute]['accelerator_type'].upper().replace("-", "_")
    machine_spec['acceleratorCount'] = config['compute'][args.compute]['n_accelerators']
    job_config_file["workerPoolSpecs"][0]['machineSpec'] = machine_spec

    scheduling = {}
    scheduling["strategy"] = "STANDARD" # STANDARD, SPOT
    job_config_file["scheduling"] = scheduling
    max_num_tokens = int(inference_batch_size * (sequence_length - max_new_tokens) * 0.8)
    env_variables = []
    env_variables.append({'name': 'AIP_STORAGE_URI', 'value': model_bucket})
    env_variables.append({'name': 'HF_BUCKET', 'value': hf_bucket})
    env_variables.append({'name': 'MAX_MODEL_LENGTH', 'value': str(sequence_length)})
    env_variables.append({'name': 'MAX_INPUT_TOKENS', 'value': str(sequence_length - max_new_tokens)})
    env_variables.append({'name': 'MAX_BATCH_SIZE', 'value': str(inference_batch_size)})
    env_variables.append({'name': 'MAX_NUM_TOKENS', 'value': str(max_num_tokens)})
    env_variables.append({'name': 'tensor_parallel', 'value': str(tp)})
    job_config_file["workerPoolSpecs"][0]['containerSpec']['env'] = env_variables

    tag = f'europe-west3-docker.pkg.dev/{project["id"]}/{image_config["repository-name"]}/{image_config["image-name"]}:latest'
    job_config_file["workerPoolSpecs"][0]['containerSpec']['imageUri'] = tag

    with open(job_config_file_path, 'w+') as ff:
        yaml.dump(job_config_file, ff)

    shell_string = f'gcloud ai custom-jobs create \
--region europe-west2 \
--display-name=trtllm-engine-build \
--config={job_config_file_path}'

    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
