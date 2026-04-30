#!/bin/bash
# Merge-only run: assumes Colab has produced the LoRA adapter at
# gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/adapter/.
# Recreates GKE, runs merge Job, collects evidence, tears down.

set -u
set -o pipefail

PROJECT_ID="ml-cloud-hw2-250147"
REGION="us-central1"
CLUSTER="ftrag-cluster"
REPO_ROOT="/Users/stl-liang/Desktop/ftRAG"
EVIDENCE_DIR="${REPO_ROOT}/docs/cloud/cloud_evidence"
STATUS_FILE=/tmp/merge_only.status

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
trap teardown EXIT

set_status "===== START merge-only run ====="

# --- Sanity check: adapter exists in GCS ---
ADAPTER_GCS="gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/adapter"
set_status "Checking adapter exists at ${ADAPTER_GCS}"
ADAPTER_LISTING=$(gcloud storage ls "${ADAPTER_GCS}/" 2>&1 || true)
if [ -z "${ADAPTER_LISTING}" ] || echo "${ADAPTER_LISTING}" | grep -q "not found\|matched no"; then
  set_status "FAIL: adapter not found at ${ADAPTER_GCS}/. Did Colab finish + upload?"
  trap - EXIT
  exit 1
fi
echo "${ADAPTER_LISTING}" | tee -a "${STATUS_FILE}"

# --- Step 1: Recreate cluster ---
set_status "STEP 1: recreating private cluster ${CLUSTER}"
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
  set_status "FAIL step 2"
  exit 1
}

# --- Step 3: Apply ServiceAccount + merge Job ---
set_status "STEP 3: applying ServiceAccount + merge Job"
kubectl apply -f cloud/k8s/serviceaccount.yaml
kubectl apply -f cloud/k8s/merge-job-colab.yaml

# --- Step 4: Wait for merge pod to schedule ---
set_status "STEP 4: waiting up to 6 min for merge pod to schedule"
SCHEDULED=0
for i in $(seq 1 12); do
  sleep 30
  STATE=$(kubectl get pods -l job-name=lora-merge -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  set_status "  schedule check ${i}: state=${STATE}"
  case "${STATE}" in
    Running|Succeeded)
      SCHEDULED=1; break;;
    Failed)
      kubectl logs -l job-name=lora-merge -c merger --tail=300 > "${EVIDENCE_DIR}/merge_pod_FAILED_logs.txt" 2>&1
      set_status "FAIL step 4: merge pod Failed"; exit 1;;
  esac
done
[ "${SCHEDULED}" -eq 1 ] || { set_status "FAIL step 4: did not schedule"; exit 1; }

# --- Step 5: Wait for merge to complete ---
set_status "STEP 5: waiting up to 25 min for merge to complete"
kubectl wait --for=condition=complete --timeout=25m job/lora-merge || {
  kubectl logs -l job-name=lora-merge -c merger --tail=500 > "${EVIDENCE_DIR}/merge_pod_TIMEOUT_logs.txt" 2>&1
  set_status "FAIL step 5: merge timeout"
  exit 1
}

# --- Step 6: Collect evidence ---
set_status "STEP 6: collecting merge evidence"
kubectl logs -l job-name=lora-merge -c merger > "${EVIDENCE_DIR}/merge_pod_logs.txt" 2>&1 || true
kubectl describe job lora-merge > "${EVIDENCE_DIR}/merge_job_describe.txt" 2>&1 || true
kubectl get job lora-merge -o yaml > "${EVIDENCE_DIR}/merge_job.yaml" 2>&1 || true
gcloud storage ls -r gs://ftrag-merged-ml-cloud-hw2-250147/llama3_merged_colab/ \
  > "${EVIDENCE_DIR}/merged_bucket_listing.txt" 2>&1 || true

set_status "===== MERGE COMPLETE ====="
set_status "Merged model written to gs://ftrag-merged-ml-cloud-hw2-250147/llama3_merged_colab/"
set_status "Evidence in ${EVIDENCE_DIR}/"
exit 0
