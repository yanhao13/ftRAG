# Runbook: deploy the FinAgentBench LoRA reranker on GKE

This is the actual sequence of commands to run, in order, to take the existing
local `lora_reranker.py` to a real cloud run on GKE Autopilot. It assumes you
already have `gcloud`, `kubectl`, and `gsutil` installed (you do — confirmed
with `gcloud --version` showing SDK 558.0.0).

Replace `PROJECT_ID` everywhere it appears (or set the env var below and use it).

## 0. One-time variables

```bash
export PROJECT_ID="REPLACE_ME"        # e.g. ftrag-term-project
export REGION="us-central1"
export CLUSTER="ftrag-cluster"
export REPO="ftrag"
export IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO}/ftrag-trainer:latest"
export SA_NAME="ftrag-gke"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
```

## 1. Set up project + billing (BLOCKER 1)

You need a GCP project with billing enabled. If you don't have one, run:

```bash
gcloud billing accounts list
# Pick a billing account ID. If empty, you need to create one in the console
# (cloud.google.com/billing) before continuing.

gcloud projects create "${PROJECT_ID}" --name="ftRAG Term Project"
gcloud config set project "${PROJECT_ID}"
gcloud billing projects link "${PROJECT_ID}" \
  --billing-account=YOUR_BILLING_ACCOUNT_ID
```

Verify:

```bash
gcloud billing projects describe "${PROJECT_ID}"
# Look for: billingEnabled: true
```

If `billingEnabled` is not `true`, stop here and fix billing first.

## 2. Enable APIs

```bash
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}"
```

## 3. Request L4 quota (BLOCKER 2 — start now, runs in background)

Go to the console: https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID

Search for and request limit `1` for each:

- `NVIDIA L4 GPUs` (region: us-central1)
- `Preemptible NVIDIA L4 GPUs` (region: us-central1)

Justification: "Class project, single-GPU LoRA fine-tuning on Llama-3.2-1B,
~30 minute job, FinAgentBench benchmark."

Approval is often within minutes for student accounts; worst case 1-3 business
days. **If denied or slow, use the T4 manifest in step 9** — T4 quota is usually
already available.

## 4. Create GCS buckets

```bash
gcloud storage buckets create \
  "gs://ftrag-data-${PROJECT_ID}" \
  "gs://ftrag-adapters-${PROJECT_ID}" \
  "gs://ftrag-merged-${PROJECT_ID}" \
  "gs://ftrag-cache-${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

The `${PROJECT_ID}` suffix avoids global-namespace collisions on `ftrag-data` etc.

Then update the four `bucketName:` lines in each k8s manifest under `cloud/k8s/`
to match the actual bucket names you just created. One quick sed:

```bash
for f in cloud/k8s/*.yaml; do
  sed -i.bak \
    -e "s|bucketName: ftrag-data\$|bucketName: ftrag-data-${PROJECT_ID}|" \
    -e "s|bucketName: ftrag-adapters\$|bucketName: ftrag-adapters-${PROJECT_ID}|" \
    -e "s|bucketName: ftrag-merged\$|bucketName: ftrag-merged-${PROJECT_ID}|" \
    -e "s|bucketName: ftrag-cache\$|bucketName: ftrag-cache-${PROJECT_ID}|" \
    "$f"
  rm -f "${f}.bak"
done
```

## 5. Upload data subset (~1.95 GB)

```bash
gsutil cp data/document_ranking_kaggle_dev.jsonl  "gs://ftrag-data-${PROJECT_ID}/"
gsutil cp data/document_ranking_kaggle_eval.jsonl "gs://ftrag-data-${PROJECT_ID}/"
gsutil cp data/chunk_ranking_kaggle_eval.jsonl    "gs://ftrag-data-${PROJECT_ID}/"
gsutil cp data/chunk_ranking_kaggle_dev.jsonl.part-00 \
  "gs://ftrag-data-${PROJECT_ID}/chunk_ranking_kaggle_dev.jsonl"
```

The default `--chunk-query-limit=20` only touches the first ~20 rows of the
chunk dev file, so part-00 alone is enough. Skipping parts -01 and -02 saves
~2.8 GB of upload.

## 6. Build the training image (Cloud Build)

```bash
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}"

gcloud builds submit --tag="${IMAGE}" --project="${PROJECT_ID}" .
```

Cloud Build runs remotely (no local Docker daemon needed). ~5 minutes,
~$0.05 in Cloud Build credits (most accounts have generous free tier).

## 7. Create GKE Autopilot cluster

```bash
gcloud container clusters create-auto "${CLUSTER}" \
  --region="${REGION}" \
  --release-channel=regular
```

~5-10 min provisioning. Cluster control plane: ~$0.10/hr while it exists.

```bash
gcloud container clusters get-credentials "${CLUSTER}" --region="${REGION}"
kubectl get nodes  # should return empty until a Job is scheduled
```

## 8. Set up Workload Identity for GCS access

```bash
# Create the GCP service account
gcloud iam service-accounts create "${SA_NAME}" --project="${PROJECT_ID}"

# Grant it Storage access on the four buckets
for B in data adapters merged cache; do
  gcloud storage buckets add-iam-policy-binding \
    "gs://ftrag-${B}-${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"
done

# Allow the GKE k8s SA to impersonate the GCP SA
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/ftrag-sa]"

# Update the k8s ServiceAccount annotation, then apply
sed -i.bak "s|PROJECT_ID|${PROJECT_ID}|g" cloud/k8s/serviceaccount.yaml
rm -f cloud/k8s/serviceaccount.yaml.bak
kubectl apply -f cloud/k8s/serviceaccount.yaml
```

## 9. Apply image substitutions, then run training Job

The manifests have `image: us-central1-docker.pkg.dev/PROJECT_ID/...` placeholder.
Substitute and apply:

```bash
sed -i.bak "s|PROJECT_ID|${PROJECT_ID}|g" cloud/k8s/train-job-l4.yaml \
                                          cloud/k8s/train-job-t4.yaml \
                                          cloud/k8s/merge-job.yaml
rm -f cloud/k8s/*.bak
```

Pick one of the two GPU manifests:

```bash
# If L4 quota is approved:
kubectl apply -f cloud/k8s/train-job-l4.yaml

# Or fall back to T4 (no quota request usually needed):
kubectl apply -f cloud/k8s/train-job-t4.yaml
```

Watch progress:

```bash
kubectl get jobs
kubectl get pods -l job-name=lora-train-l4   # or lora-train-t4
kubectl logs -f -l job-name=lora-train-l4    # tail training output
```

Pod scheduling takes 3-5 min on Autopilot (it has to provision a GPU node).
Training itself: ~15-30 min on L4, ~30-60 min on T4.

When the Job's `COMPLETIONS` shows `1/1`, verify outputs:

```bash
gsutil ls -r "gs://ftrag-adapters-${PROJECT_ID}/llama3_lora_reranker/"
# should show: adapter/, metrics.json
gsutil cat "gs://ftrag-adapters-${PROJECT_ID}/llama3_lora_reranker/metrics.json"
```

## 10. Run the merge Job

```bash
kubectl apply -f cloud/k8s/merge-job.yaml
kubectl logs -f -l job-name=lora-merge
```

~5 min. CPU-only, very cheap. When done:

```bash
gsutil ls "gs://ftrag-merged-${PROJECT_ID}/llama3_merged/"
# should show: model.safetensors, config.json, tokenizer files,
# merge_manifest.json
```

## 11. Download artifacts to your laptop

```bash
mkdir -p outputs/cloud
gsutil -m cp -r \
  "gs://ftrag-adapters-${PROJECT_ID}/llama3_lora_reranker/" \
  outputs/cloud/
gsutil -m cp -r \
  "gs://ftrag-merged-${PROJECT_ID}/llama3_merged/" \
  outputs/cloud/
```

`outputs/cloud/llama3_lora_reranker/metrics.json` is the LoRA reranker's own
base-vs-tuned metrics. Screenshot for slides if needed.

## 12. Tear down to stop billing

```bash
kubectl delete -f cloud/k8s/train-job-l4.yaml --ignore-not-found
kubectl delete -f cloud/k8s/train-job-t4.yaml --ignore-not-found
kubectl delete -f cloud/k8s/merge-job.yaml --ignore-not-found
gcloud container clusters delete "${CLUSTER}" --region="${REGION}" --quiet
```

The buckets keep their data and cost pennies per month. Delete them too if
you want a complete cleanup:

```bash
gsutil -m rm -r \
  "gs://ftrag-data-${PROJECT_ID}/**" \
  "gs://ftrag-adapters-${PROJECT_ID}/**" \
  "gs://ftrag-merged-${PROJECT_ID}/**" \
  "gs://ftrag-cache-${PROJECT_ID}/**"
gcloud storage buckets delete \
  "gs://ftrag-data-${PROJECT_ID}" \
  "gs://ftrag-adapters-${PROJECT_ID}" \
  "gs://ftrag-merged-${PROJECT_ID}" \
  "gs://ftrag-cache-${PROJECT_ID}"
```

## Cost estimate (us-central1, as of 2026)

| Step | Resource | Time | Cost |
| --- | --- | --- | --- |
| Cloud Build | 1 worker | 5 min | ~$0.05 |
| GCS storage (4 buckets) | ~5 GB Standard | 1 month | ~$0.10 |
| GKE Autopilot cluster | control plane | duration of run | ~$0.10/hr |
| L4 Spot training | 1 GPU, 8 vCPU, 32 GB | 30 min | ~$0.20 |
| T4 Spot training (fallback) | 1 GPU, 4 vCPU, 16 GB | 60 min | ~$0.15 |
| Merge job | 4 vCPU, 16 GB CPU-only | 5 min | ~$0.02 |
| **Realistic first run** | — | ~1 hr end-to-end | **~$0.50-2** |

## What this run produces vs the headline metrics

The cloud Job produces `outputs/cloud/llama3_lora_reranker/metrics.json` with
the LoRA reranker's own `nDCG@5 / MAP@5 / MRR@5`. These numbers will differ
from the local `outputs/chunk_ltr_5000/metrics_best_ensemble.json` headline
(0.432 / 0.391 / 0.640) because that file is the *supervised-ranker ensemble*
output from `chunk_best_ensemble.py`, which doesn't use the LoRA model at all.

To reproduce the ensemble headline on cloud you'd also need to upload
`outputs/chunk_ltr_5000/` and run `chunk_best_ensemble.py` as a separate CPU
Job, but the result would be byte-identical to the local file (same input,
deterministic) so it's not normally worth doing.
