#!/bin/bash
# Overnight script: recreate cluster, run train + merge on cloud, tear down.
# Logs to /tmp/finish_overnight.log
# Safe to interrupt; cluster teardown runs in trap.

set -u
set -o pipefail

PROJECT_ID="ml-cloud-hw2-250147"
REGION="us-central1"
CLUSTER="ftrag-cluster"
REPO_ROOT="/Users/stl-liang/Desktop/ftRAG"
EVIDENCE_DIR="${REPO_ROOT}/docs/cloud/cloud_evidence"
STATUS_FILE=/tmp/finish_overnight.status

cd "${REPO_ROOT}"

mkdir -p "${EVIDENCE_DIR}"

set_status() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${STATUS_FILE}"
}

teardown() {
  set_status "TEARDOWN: deleting cluster ${CLUSTER}"
  gcloud container clusters delete "${CLUSTER}" --region="${REGION}" --quiet || true
  set_status "TEARDOWN: complete"
}

# If anything fails after cluster creation, tear down on exit.
trap teardown EXIT

set_status "===== START overnight run ====="

# --- Step 1: Recreate cluster (private, due to org policy on external IPs) ---
set_status "STEP 1: recreating private cluster ${CLUSTER} in ${REGION} (~7 min)"
gcloud container clusters create-auto "${CLUSTER}" \
  --region="${REGION}" \
  --release-channel=regular \
  --enable-private-nodes \
  --master-ipv4-cidr=172.16.0.32/28 \
  --enable-master-authorized-networks \
  --master-authorized-networks=0.0.0.0/0 || {
    set_status "FAIL step 1: cluster creation"
    exit 1
  }

# --- Step 2: Get credentials ---
set_status "STEP 2: getting kubectl credentials"
gcloud container clusters get-credentials "${CLUSTER}" --region="${REGION}" || {
  set_status "FAIL step 2: get-credentials"
  exit 1
}

# --- Step 3: Apply ServiceAccount ---
set_status "STEP 3: applying ServiceAccount"
kubectl apply -f cloud/k8s/serviceaccount.yaml || {
  set_status "FAIL step 3: serviceaccount"
  exit 1
}

# --- Step 4: Apply train job (on-demand L4) ---
set_status "STEP 4: applying L4 on-demand train job"
kubectl apply -f cloud/k8s/train-job-l4-ondemand.yaml || {
  set_status "FAIL step 4: train job apply"
  exit 1
}

# --- Step 5: Wait up to 12 min for pod to schedule ---
set_status "STEP 5: waiting up to 12 min for train pod to schedule"
SCHEDULED=0
for i in $(seq 1 24); do
  sleep 30
  STATE=$(kubectl get pods -l job-name=lora-train-l4 -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  set_status "  schedule check ${i}: state=${STATE}"
  case "${STATE}" in
    Running|Succeeded)
      SCHEDULED=1
      break
      ;;
    Failed)
      kubectl describe pods -l job-name=lora-train-l4 > "${EVIDENCE_DIR}/train_pod_FAILED_describe.txt" 2>&1
      kubectl logs -l job-name=lora-train-l4 -c trainer --tail=300 > "${EVIDENCE_DIR}/train_pod_FAILED_logs.txt" 2>&1
      set_status "FAIL step 5: pod entered Failed"
      exit 1
      ;;
  esac
done

if [ "${SCHEDULED}" -ne 1 ]; then
  kubectl describe pods -l job-name=lora-train-l4 > "${EVIDENCE_DIR}/train_pod_PENDING_describe.txt" 2>&1
  set_status "FAIL step 5: pod did not schedule within 12 min (likely GPU capacity)"
  exit 1
fi

# --- Step 6: Wait up to 75 min for train job to complete ---
set_status "STEP 6: train pod scheduled. Waiting up to 75 min for completion."
kubectl wait --for=condition=complete --timeout=75m job/lora-train-l4 || {
  kubectl describe job lora-train-l4 > "${EVIDENCE_DIR}/train_job_TIMEOUT_describe.txt" 2>&1
  kubectl logs -l job-name=lora-train-l4 -c trainer --tail=500 > "${EVIDENCE_DIR}/train_pod_TIMEOUT_logs.txt" 2>&1
  set_status "FAIL step 6: train job did not complete in 75 min"
  exit 1
}
set_status "STEP 6: train job COMPLETE"

# --- Step 7: Save train artifacts as evidence ---
set_status "STEP 7: collecting train evidence"
kubectl logs -l job-name=lora-train-l4 -c trainer > "${EVIDENCE_DIR}/train_pod_logs.txt" 2>&1 || true
kubectl describe job lora-train-l4 > "${EVIDENCE_DIR}/train_job_describe.txt" 2>&1 || true
kubectl get job lora-train-l4 -o yaml > "${EVIDENCE_DIR}/train_job.yaml" 2>&1 || true
gcloud storage ls -r gs://ftrag-adapters-ml-cloud-hw2-250147/ > "${EVIDENCE_DIR}/adapters_bucket_listing.txt" 2>&1 || true

# Best-effort metrics pull (filename varies)
gcloud storage cp \
  gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker/metrics.json \
  "${EVIDENCE_DIR}/lora_train_metrics.cloud.json" 2>&1 || true

# --- Step 8: Apply merge job ---
set_status "STEP 8: applying merge job"
kubectl apply -f cloud/k8s/merge-job.yaml || {
  set_status "FAIL step 8: merge job apply"
  exit 1
}

# --- Step 9: Wait for merge schedule ---
set_status "STEP 9: waiting up to 8 min for merge pod to schedule"
SCHEDULED=0
for i in $(seq 1 16); do
  sleep 30
  STATE=$(kubectl get pods -l job-name=lora-merge -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  set_status "  merge schedule check ${i}: state=${STATE}"
  case "${STATE}" in
    Running|Succeeded)
      SCHEDULED=1
      break
      ;;
    Failed)
      kubectl describe pods -l job-name=lora-merge > "${EVIDENCE_DIR}/merge_pod_FAILED_describe.txt" 2>&1
      kubectl logs -l job-name=lora-merge -c merger --tail=300 > "${EVIDENCE_DIR}/merge_pod_FAILED_logs.txt" 2>&1
      set_status "FAIL step 9: merge pod entered Failed"
      exit 1
      ;;
  esac
done

if [ "${SCHEDULED}" -ne 1 ]; then
  set_status "FAIL step 9: merge pod did not schedule"
  exit 1
fi

# --- Step 10: Wait for merge to complete ---
set_status "STEP 10: waiting up to 25 min for merge to complete"
kubectl wait --for=condition=complete --timeout=25m job/lora-merge || {
  kubectl describe job lora-merge > "${EVIDENCE_DIR}/merge_job_TIMEOUT_describe.txt" 2>&1
  kubectl logs -l job-name=lora-merge -c merger --tail=500 > "${EVIDENCE_DIR}/merge_pod_TIMEOUT_logs.txt" 2>&1
  set_status "FAIL step 10: merge timeout"
  exit 1
}

# --- Step 11: Save merge evidence ---
set_status "STEP 11: collecting merge evidence"
kubectl logs -l job-name=lora-merge -c merger > "${EVIDENCE_DIR}/merge_pod_logs.txt" 2>&1 || true
kubectl describe job lora-merge > "${EVIDENCE_DIR}/merge_job_describe.txt" 2>&1 || true
kubectl get job lora-merge -o yaml > "${EVIDENCE_DIR}/merge_job.yaml" 2>&1 || true
gcloud storage ls -r gs://ftrag-merged-ml-cloud-hw2-250147/ > "${EVIDENCE_DIR}/merged_bucket_listing.txt" 2>&1 || true

set_status "===== ALL CLOUD STEPS COMPLETED SUCCESSFULLY ====="
set_status "Train + merge produced artifacts in:"
set_status "  gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker/"
set_status "  gs://ftrag-merged-ml-cloud-hw2-250147/llama3_merged/"
set_status "Evidence saved to: ${EVIDENCE_DIR}"

# trap will run teardown on exit
exit 0
