#!/bin/bash
set -e

HF_BUCKET=${HF_BUCKET:?HF_BUCKET env var is required}
ENGINE_DIR=${LOCAL_MODEL_DIR}/tensorrt_llm/1
dTYPE="bfloat16"
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}

echo "--- Environment variables ---"
for var in HF_BUCKET AIP_STORAGE_URI MAX_MODEL_LENGTH MAX_INPUT_TOKENS MAX_BATCH_SIZE LOCAL_CHECKPOINT_DIR LOCAL_MODEL_DIR LOCAL_HF_DIR; do
  echo "env variable ${var}=${!var:-<unset>}"
done
echo "-----------------------------"

mkdir -p ${LOCAL_MODEL_DIR}
mkdir -p ${LOCAL_HF_DIR}

echo "--- Build triton folder structure ---"
#INFLIGHT_DIR=$(find / -type d -name "inflight_batcher_llm" 2>/dev/null | head -1)
#INFLIGHT_DIR="app/triton_backend/all_models/inflight_batcher_llm"

INFLIGHT_DIR=$(find / -type d -path "*/all_models/inflight_batcher_llm" 2>/dev/null | head -1)

if [ -z "$INFLIGHT_DIR" ]; then
  echo "ERROR: inflight_batcher_llm not found in container. Aborting."
  exit 1
fi
echo "Found at: ${INFLIGHT_DIR}"
cp -r ${INFLIGHT_DIR}/* ${LOCAL_MODEL_DIR}
echo "-----------------------------"
cp /config.yaml ${LOCAL_MODEL_DIR}

echo "--- Downloading HF weights from GCS ---"
gcloud storage cp -r ${HF_BUCKET}/* ${LOCAL_HF_DIR}
echo "-----------------------------"

echo "--- Converting HF checkpoint to TRT-LLM format ---"
CONVERT_SCRIPT=$(find / -type f -name "convert_checkpoint.py" -path "*/llama/*" 2>/dev/null | head -1)
if [ -z "$CONVERT_SCRIPT" ]; then
  echo "ERROR: llama convert_checkpoint.py not found. Aborting."
  exit 1
fi
echo "Found at: ${CONVERT_SCRIPT}"
python3 ${CONVERT_SCRIPT} \
    --model_dir ${LOCAL_HF_DIR} \
    --output_dir ${LOCAL_CHECKPOINT_DIR} \
    --dtype ${dTYPE} \
    --tp_size ${TENSOR_PARALLEL}
echo "-----------------------------"

echo "--- Building TRT-LLM engines ---"
trtllm-build \
    --checkpoint_dir ${LOCAL_CHECKPOINT_DIR} \
    --remove_input_padding enable \
    --gpt_attention_plugin auto \
    --gemm_plugin auto \
    --output_dir ${ENGINE_DIR} \
    --kv_cache_type paged \
    --use_paged_context_fmha enable \
    --context_fmha enable \
    --multiple_profiles enable \
    --max_input_len ${MAX_INPUT_TOKENS} \
    --max_seq_len ${MAX_MODEL_LENGTH} \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --max_num_tokens ${MAX_NUM_TOKENS}
echo "-----------------------------"

echo "--- Uploading engines and config files to GCS ---"
# Upload engines, replacing the raw HF files
gcloud storage cp -r ${LOCAL_MODEL_DIR}/* ${AIP_STORAGE_URI}/
echo "-----------------------------"

echo "--- Cleaning folders ---"
rm -rf ${LOCAL_HF_DIR} ${LOCAL_CHECKPOINT_DIR}
echo "--- Done ---"