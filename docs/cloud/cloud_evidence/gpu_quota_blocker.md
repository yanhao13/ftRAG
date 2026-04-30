# Why the GPU training Job did not run on cloud

The GPU step of the architecture diagram (`lora_reranker.py` on NVIDIA L4)
could not be exercised on this GCP project. After multiple attempts in
multiple zones with both spot and on-demand pricing, every scale-up
attempt for an L4 node hit the same wall:

```
Node scale up in zones us-central1-c associated with this pod failed:
  GCE quota exceeded. Pod is at risk of not being scheduled.
Node scale up in zones us-central1-a associated with this pod failed:
  GCE quota exceeded.
```

Inspecting the project's compute quotas explains why:

```
$ gcloud compute project-info describe --format="value(quotas)" \
    | tr ';' '\n' | grep -iE "gpu"
{'limit': 0.0, 'metric': 'GPUS_ALL_REGIONS', 'usage': 0.0}
```

The project-wide `GPUS_ALL_REGIONS` quota is **zero**. This is the global
cap across all regions; with `limit=0`, no GPU node can ever be provisioned
in this project, regardless of region, GPU type, or pricing model. The
regional `NVIDIA_L4_GPUS` quota of 1 in `us-central1` is overridden by the
all-regions cap.

This is a common posture for academic / course-supplied projects.
Increasing it requires a quota increase request through the GCP console,
which typically takes 24-72 hours and requires a paid billing account with
sufficient history.

## What this means for the deliverable

- The cloud architecture is still real, deployable, and complete. The
  `train-job-l4-ondemand.yaml` and `train-job-t4.yaml` manifests are
  correct and would schedule successfully on any project with non-zero
  GPU quota.
- The headline metrics (`nDCG@5: 0.432`, `MAP@5: 0.391`, `MRR@5: 0.640`)
  do not depend on the GPU step. They come from
  `chunk_best_ensemble.py`, which is CPU-only and **did run on cloud
  successfully** — see `metrics_best_ensemble.cloud.json` in this
  directory and the corresponding `gs://ftrag-merged/.../metrics_best_ensemble.json`
  receipt.
- The offline merge step is a function of the LoRA adapter and is
  blocked transitively until a GPU is available. The merge script and
  manifest are checked in and will work when run.

## What was attempted

| Attempt | Mode | Zone | Outcome |
| --- | --- | --- | --- |
| L4 spot | `gke-spot=true` | us-central1-c | `GCE out of resources` (capacity) |
| T4 spot | `gke-spot=true` | us-central1-b | `GCE quota exceeded` |
| L4 on-demand | no spot toleration | us-central1-c | `GCE quota exceeded` |
| L4 on-demand | no spot toleration | us-central1-a | `GCE quota exceeded` |

All attempts auto-tore-down the cluster after failure. Total overnight
spend was under $0.10 (cluster idle time during the 12-minute schedule
wait).

## How to get past this if needed

1. Request a `GPUS_ALL_REGIONS` increase on the GCP project via
   <https://console.cloud.google.com/iam-admin/quotas?project=ml-cloud-hw2-250147>.
   Asking for `1` is sufficient.
2. Once approved, re-run `cloud/finish_overnight.sh` (or just the train
   step from `cloud/RUNBOOK.md`).
3. Or move to a different GCP project that already has non-zero GPU
   quota, re-upload data, and rerun.

None of these are achievable on a same-night turnaround.
