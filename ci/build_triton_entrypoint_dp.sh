#!/bin/bash
set -e

# ─── Paths ────────────────────────────────────────────────────────────────────

HF_BUCKET=${HF_BUCKET:?HF_BUCKET env var is required}
TRITON_MAX_BATCH_SIZE=${TRITON_MAX_BATCH_SIZE:-4}
INSTANCE_COUNT=${INSTANCE_COUNT:-1}
MAX_QUEUE_DELAY_MS=${MAX_QUEUE_DELAY_MS:-0}
MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-0}
LOGITS_DATATYPE=${LOGITS_DATATYPE:-TYPE_BF16}
DECOUPLED_MODE=${DECOUPLED_MODE:-false}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
NUM_GPUS=${NUM_GPUS:-2}

mkdir -p ${LOCAL_MODEL_DIR}
mkdir -p ${LOCAL_HF_DIR}

echo "--- Authenticate ---"
gcloud config set account ${SA_EMAIL}

echo "--- Environment variables ---"
for var in HF_BUCKET AIP_STORAGE_URI LOCAL_MODEL_DIR LOCAL_HF_DIR; do
  echo "env variable ${var}=${!var:-<unset>}"
done
echo "-----------------------------"

# ─── Download model repo (config templates + engine) from Vertex AI GCS ───────
echo "Downloading model repo from ${AIP_STORAGE_URI}..."
gcloud storage cp -r ${AIP_STORAGE_URI}/* ${LOCAL_MODEL_DIR}/

# ─── Download tokenizer files only (skip safetensors weights) ─────────────────
echo "Downloading tokenizer files from ${HF_BUCKET}..."
gcloud storage cp \
  "${HF_BUCKET}/tokenizer.json" \
  "${HF_BUCKET}/tokenizer_config.json" \
  "${HF_BUCKET}/special_tokens_map.json" \
  "${HF_BUCKET}/model.safetensors.index.json" \
  "${HF_BUCKET}/config.json" \
  ${LOCAL_HF_DIR}/

# ─── Fill config templates ─────────────────────────────────────────────────────
FILL_TEMPLATE_SCRIPT=/app/tools/fill_template.py
ENGINE_DIR=${LOCAL_MODEL_DIR}/tensorrt_llm/1

echo "Filling config templates..."

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/preprocessing/config.pbtxt tokenizer_dir:${LOCAL_HF_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${NUM_GPUS}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,gpu_device_ids:"0;1",triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_beam_width:1,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:${LOGITS_DATATYPE},logits_datatype:${LOGITS_DATATYPE},enable_chunked_context:true,kv_cache_free_gpu_mem_fraction:0.9,batch_scheduler_policy:max_utilization

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/postprocessing/config.pbtxt tokenizer_dir:${LOCAL_HF_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${NUM_GPUS}

#python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${TRITON_MAX_BATCH_SIZE},accumulate_tokens:false,logits_datatype:${LOGITS_DATATYPE}

# ─── Launch Triton ─────────────────────────────────────────────────────────────
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_P2P_DISABLE=0
#export NCCL_SHM_DISABLE=0
#export NCCL_ALGO=Tree  # Sometimes performs better over PCIe than Ring

echo "Starting Triton Inference Server (Data Parallel x2)..."
/opt/tritonserver/bin/tritonserver \
  --model-repository=${LOCAL_MODEL_DIR} \
  --allow-http=true \
  --log-verbose=1 \
  --multi-model \
  --allow-vertex-ai=false \
  --http-port=${AIP_HTTP_PORT} \
  --model-control-mode=explicit \
  --load-model=preprocessing \
  --load-model=tensorrt_llm \
  --load-model=postprocessing \
  --load-model=ensemble \
  --disable-auto-complete-config \
  --backend-config=python,shm-region-prefix-name=prefix0_