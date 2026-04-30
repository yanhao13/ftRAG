#!/usr/bin/env bash
# Polls GCS every 60s waiting for the Colab notebook to finish uploading.
# Marker: gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/metrics.json
# When detected, writes ./cloud/.colab_done and exits 0.

set -u
ADAPTER_URI="gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/metrics.json"
MARKER="cloud/.colab_done"
LOG="cloud/watch_colab_done.log"

cd "$(dirname "$0")/.."  # repo root

rm -f "$MARKER"
echo "[$(date '+%H:%M:%S')] watching $ADAPTER_URI" | tee -a "$LOG"

while true; do
  if gcloud storage ls "$ADAPTER_URI" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] DETECTED metrics.json in GCS" | tee -a "$LOG"
    gcloud storage ls -r "gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/" >> "$LOG" 2>&1
    touch "$MARKER"
    exit 0
  fi
  echo "[$(date '+%H:%M:%S')] not yet, sleeping 60s" >> "$LOG"
  sleep 60
done
