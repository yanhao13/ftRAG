# Colab fallback path for LoRA training

## Why we're here

The GCP project `ml-cloud-hw2-250147` (and the other 8 projects on the same
`ts3479@columbia.edu` account that we checked) all have
`GPUS_ALL_REGIONS: limit=0` — a hard project-wide GPU cap of zero.
Increasing this requires a quota request that takes 24-72 hours and often
isn't approved for educational projects.

The CPU pieces of the architecture (data prep, ensemble scoring, offline
merge) all ran successfully on GKE; only the GPU training step is blocked.

This folder lets us complete the GPU training step on **Google Colab**
(free T4 GPU), then pipe the resulting LoRA adapter back into the existing
GCS bucket so the rest of the GKE pipeline stays unchanged.

The resulting story: **trained on Colab, merged on GKE, served on GKE**.
The training-pipeline diagram needs one box updated to say "Colab GPU
notebook" instead of "GKE Pod" to be honest.

## How to run it (about 5 minutes of setup, 30-50 min of training)

1. Open <https://colab.research.google.com>.
2. **File -> Upload notebook**, select
   `cloud/colab/lora_train_colab.ipynb` from this repo on your laptop.
3. **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU**,
   click Save.
4. **Runtime -> Run all**. About 5 cells will run; the second will
   pop up a Google OAuth dialog — sign in with `ts3479@columbia.edu`
   so Colab can read/write the GCS buckets you already own.
5. The training cell prints a tqdm progress bar. Don't close the tab.
   Total wall time: 30-50 min on a free Colab T4.
6. Last cell pushes everything to
   `gs://ftrag-adapters-ml-cloud-hw2-250147/llama3_lora_reranker_colab/`.
7. Come back here and tell the agent "colab run finished". The agent
   will pull the metrics, update the architecture diagram, run the
   offline merge step (no GPU needed), and refresh the slide deliverable.

## What the notebook produces

In the GCS bucket `ftrag-adapters-ml-cloud-hw2-250147` under
`llama3_lora_reranker_colab/`:

- `adapter/` — the LoRA adapter weights and tokenizer.
- `metrics.json` — base vs LoRA-fine-tuned `nDCG@5/MAP@5/MRR@5` on a
  500-query chunk validation split (directly comparable to the
  ensemble's 0.432).
- `validation_rankings.{csv,jsonl}` — per-query LoRA rankings.

The `metrics.json` is the answer to "what did the LoRA fine-tuning
actually buy us" — measured honestly on the same 500-query split the
paper's Table 3 uses.

## Failure modes and what to do

- **Colab gives you a non-T4 GPU** (sometimes V100 or A100 free): great,
  it just trains faster. No code change needed.
- **Colab gives you no GPU** (busy hour): the runtime change dialog
  shows "None". Wait 10 minutes and retry, or use Colab Pro.
- **OOM during training**: edit the `--batch-size=2` argument in the
  training cell to `--batch-size=1` and rerun.
- **Auth fails** in cell 2: make sure you're signing in with
  `ts3479@columbia.edu` (the account that owns the buckets). If it
  still fails, check that the GCS objects are not in a different
  project under the same account.
- **Runtime disconnect**: Colab free disconnects after ~12 hours or
  90 min idle. The training cell should finish well within that. If
  it disconnects mid-run, just rerun — model loading and tokenization
  are fast; we lose only the in-progress step.

## Why not Kaggle / RunPod / Lambda?

- **Kaggle Notebooks**: free P100 GPU, no time limits during a session,
  also a fine choice. The notebook would need minor tweaks (Kaggle
  uses different auth path for GCS). Open if you have Kaggle preference.
- **RunPod / Lambda Labs / Vast.ai**: ~$0.30-0.60/hr for L4-class.
  Would work, but adds another account and a credit card.

Colab is the smallest deviation from your existing toolchain and is free.
