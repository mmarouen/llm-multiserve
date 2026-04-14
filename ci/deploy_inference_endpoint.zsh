#!/bin/zsh
set -e
export PYTHONPATH=$PYTHONPATH:.
VENV_PATH="/Users/marouenazzouz/venv311/bin/activate"
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

REGION=${region:-'europe-west3'}
UPDATE_MODEL=${update:-0}
DESCRIPTION=${description:-"3rd iteration: integration within fastapi"}
MODEL_NAME=${model:-"llama-3.2-trtllm"}
SPOT=${spot:-'False'}

echo "REGION: $REGION UPDATE_MODEL: $UPDATE_MODEL MODEL_NAME: $MODEL_NAME DESCRIPTION $DESCRIPTION SPOT: $SPOT"

source "$VENV_PATH"
if [[ "$UPDATE_MODEL" -gt 1 ]]; then

    python3 scripts/update_repository.py --model-name "$MODEL_NAME" --region "$REGION"
    echo "UPDATE REPOSITORY FINISH, WAIT 45s TO TAKE EFFECT"
    echo "---------------------------------"
    sleep 45
fi

if [[ "$UPDATE_MODEL" -gt 0 ]]; then
    python3 scripts/update_model.py --model-name "$MODEL_NAME" --description "$DESCRIPTION" --region "$REGION"
    echo "UPDATE MODEL FINISH, WAIT 45s TO TAKE EFFECT"
    echo "---------------------------------"
    sleep 45
fi

python3 scripts/update_deployment.py --model-name "$MODEL_NAME" --description "$DESCRIPTION" --region "$REGION" --spot "$SPOT"
echo "UPDATE DEPLOYMENT FINISH"
echo "---------------------------------"
