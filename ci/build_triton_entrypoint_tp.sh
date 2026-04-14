#!/bin/bash
set -e

# ─── Paths ────────────────────────────────────────────────────────────────────

HF_BUCKET=${HF_BUCKET:?HF_BUCKET env var is required}
TRITON_MAX_BATCH_SIZE=${TRITON_MAX_BATCH_SIZE:-4}
INSTANCE_COUNT=${INSTANCE_COUNT:-1}
MAX_QUEUE_DELAY_MS=${MAX_QUEUE_DELAY_MS:-0}
MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-0}
LOGITS_DATATYPE=${LOGITS_DATATYPE:-TYPE_FP32}
ENCODER_INPUT_FEATURE_DATATYPE=${ENCODER_INPUT_FEATURE_DATATYPE:-TYPE_FP16}
DECOUPLED_MODE=${DECOUPLED_MODE:-false}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}

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

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/preprocessing/config.pbtxt tokenizer_dir:${LOCAL_HF_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}

#python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:${LOGITS_DATATYPE},logits_datatype:${LOGITS_DATATYPE},enable_chunked_context:true,kv_cache_free_gpu_mem_fraction:0.9,batch_scheduler_policy:max_utilization
python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_beam_width:1,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:${ENCODER_INPUT_FEATURE_DATATYPE},enable_chunked_context:true,logits_datatype:${LOGITS_DATATYPE},kv_cache_free_gpu_mem_fraction:0.9,batch_scheduler_policy:max_utilization

python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/postprocessing/config.pbtxt tokenizer_dir:${LOCAL_HF_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}

#python3 ${FILL_TEMPLATE_SCRIPT} -i ${LOCAL_MODEL_DIR}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${TRITON_MAX_BATCH_SIZE},accumulate_tokens:false,logits_datatype:${LOGITS_DATATYPE}

# ─── Launch Triton ─────────────────────────────────────────────────────────────
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_P2P_DISABLE=0
#export NCCL_SHM_DISABLE=0
#export NCCL_ALGO=Tree  # Sometimes performs better over PCIe than Ring

echo "Starting Triton Inference Server..."
mpirun --allow-run-as-root \
  --bind-to none \
  -x NCCL_DEBUG \
  -n 1 /opt/tritonserver/bin/tritonserver \
    --model-repository=${LOCAL_MODEL_DIR} \
    --allow-http=true \
    --log-verbose=1 \
    --allow-vertex-ai=false \
    --http-port=${AIP_HTTP_PORT} \
    --model-control-mode=explicit \
    --load-model=preprocessing \
    --load-model=tensorrt_llm \
    --load-model=postprocessing \
    --load-model=ensemble \
    --disable-auto-complete-config \
    --backend-config=python,shm-region-prefix-name=prefix0_ \
  : \
  -n 1 /opt/tritonserver/bin/tritonserver \
    --model-repository=${LOCAL_MODEL_DIR} \
    --model-control-mode=explicit \
    --load-model=tensorrt_llm \
    --http-port=${AIP_HTTP_PORT} \
    --allow-http=false \
    --allow-vertex-ai=false \
    --disable-auto-complete-config \
    --backend-config=python,shm-region-prefix-name=prefix1_ \
  :
