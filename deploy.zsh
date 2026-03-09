#!/bin/zsh
set -e

VENV_PATH="/Users/marouenazzouz/venv311/bin/activate"
MODEL_VERSION=24
MODEL_NAME=llama-3.2-vllm
DESCRIPTION="tensor parallel"
source "$VENV_PATH"

python3 scripts/update_repository.py --model-version "$MODEL_VERSION" --model-name "$MODEL_NAME"
echo "UPDATE REPOSITORY FINISH, WAIT 45s TO TAKE EFFECT"
echo "---------------------------------"
sleep 45

python3 scripts/update_model.py --model-name "$MODEL_NAME" --description "$DESCRIPTION"
echo "UPDATE MODEL FINISH, WAIT 45s TO TAKE EFFECT"
echo "---------------------------------"
sleep 45

python3 scripts/update_deployment.py --model-version "$MODEL_VERSION" --model-name "$MODEL_NAME" --description "$DESCRIPTION"
echo "UPDATE DEPLOYMENT FINISH"
echo "---------------------------------"
