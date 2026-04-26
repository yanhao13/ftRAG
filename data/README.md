# Llama-3 LoRA and Chunk Ranking

This folder contains two ranking approaches for the Kaggle-style document/chunk ranking JSONL files:

- `lora_reranker.py`: a Llama-3-family LoRA sequence-classification reranker.
- `chunk_ranker.py` and `chunk_best_ensemble.py`: a faster supervised chunk-ranking baseline that produced the best local validation result.

The selected Llama base model is [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). It is small enough to LoRA-tune locally and avoids gated Meta checkpoint setup. To use an official Meta checkpoint after accepting the license and logging into Hugging Face, pass `--model-name meta-llama/Llama-3.2-1B-Instruct`.

## Install

```bash
python3 -m pip install --user -r requirements-lora.txt
```

## Llama-3 LoRA Smoke Test

```bash
python3 lora_reranker.py \
  --epochs 1 \
  --doc-query-limit 30 \
  --chunk-query-limit -1 \
  --doc-val-queries 10 \
  --chunk-val-queries 0 \
  --max-train-pairs 120 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --output-dir outputs/llama3_smoke
```

## Chunk Ranking

Train the cached supervised rankers:

```bash
python3 chunk_ranker.py \
  --train-query-limit 5000 \
  --val-queries 500 \
  --candidate-char-limit 6000 \
  --train-aux-rankers \
  --predict-eval \
  --output-dir outputs/chunk_ltr_5000 \
  --cache-dir outputs/chunk_ltr_cache
```

Materialize the best row-level ensemble:

```bash
python3 chunk_best_ensemble.py
```

The best verified chunk validation metrics are in `outputs/chunk_ltr_5000/metrics_best_ensemble.json`.

| Metric | Table 3 | Best ensemble |
| --- | ---: | ---: |
| nDCG@5 | 0.371 | 0.431975 |
| MAP@5 | 0.274 | 0.390767 |
| MRR@5 | 0.587 | 0.639567 |

The chunk dev JSONL reuses some `uuid` values across separate rows, so `chunk_best_ensemble.py` reports duplicate-safe row-level metrics. Final eval rankings are saved as:

- `outputs/chunk_ltr_5000/chunk_eval_rankings_best_ensemble.csv`
- `outputs/chunk_ltr_5000/chunk_eval_rankings_best_ensemble.jsonl`

Large raw datasets, feature caches, model joblibs, and LoRA adapters are intentionally ignored by git.
