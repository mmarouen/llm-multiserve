#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python3 serve.py --model-version ${MODEL_VERSION} --port 8081 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python3 serve.py --model-version ${MODEL_VERSION} --port 8082 &
P2=$!
echo "Waiting for workers to be ready..."
until curl -sf http://127.0.0.1:8081/health && curl -sf http://127.0.0.1:8082/health; do
    sleep 5
done
echo "Both workers ready, starting nginx"

nginx -g "daemon off;"
P3=$!

wait -n $P1 $P2 $P3
echo "A process exited unexpectedly"
kill $P1 $P2 $P3 2>/dev/null
exit 1
