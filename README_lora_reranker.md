# Llama-3 LoRA Reranker Fine-Tuning

This setup fine-tunes a Llama-3-family model as a reranker instead of doing slow generative SFT over the full prompts. It converts each query into `(question, candidate)` pairs, trains a LoRA adapter to score relevance, ranks candidates by score, and reports the screenshot metrics: `nDCG@5`, `MAP@5`, and `MRR@5`.

Chosen base model: [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). It is small enough to LoRA-tune locally and keeps the run practical compared with Llama-3 8B. Official Meta Llama repos may require gated Hugging Face access, so this default avoids that setup step.

The script disables Hugging Face Xet downloads by default (`HF_HUB_DISABLE_XET=1`) because plain HTTP was more reliable in this local environment.

## Install

```bash
python3 -m pip install --user -r requirements-lora.txt
```

## Quick Run

This keeps runtime low on a Mac by using 200 document dev rows, 20 chunk dev rows, short candidate windows, and at most 1,000 sampled training pairs.

```bash
python3 lora_reranker.py \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16
```

Outputs go to `outputs/minilm_lora_reranker/`:

- `adapter/`: the Llama LoRA adapter and tokenizer files.
- `metrics.json`: base vs fine-tuned `nDCG@5`, `MAP@5`, and `MRR@5`.
- `eval_rankings.jsonl` and `eval_rankings.csv`: rankings for the unlabeled Kaggle eval files when `--predict-eval` is used.

## Faster Smoke Test

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

## More Data

Set `--doc-query-limit 0 --chunk-query-limit 0 --max-train-pairs 0 --max-length 512 --candidate-char-limit 6000` to scan all dev rows with longer candidate context. That uses the full 4.6 GB chunk file and will take much longer.

To use an official gated Meta checkpoint after accepting the license and logging into Hugging Face, pass:

```bash
python3 lora_reranker.py --model-name meta-llama/Llama-3.2-1B-Instruct
```

## Chunk Ranking Result

The best fast chunk-ranking run uses the cached supervised rankers from `chunk_ranker.py` and materializes a row-level ensemble:

```bash
python3 chunk_best_ensemble.py
```

The verified validation metrics are written to `outputs/chunk_ltr_5000/metrics_best_ensemble.json`.

| Metric | Table 3 | Best ensemble |
| --- | ---: | ---: |
| nDCG@5 | 0.371 | 0.431975 |
| MAP@5 | 0.274 | 0.390767 |
| MRR@5 | 0.587 | 0.639567 |

This uses duplicate-safe row-level scoring because the chunk dev JSONL reuses some `uuid` values across separate rows. Eval rankings are written to:

- `outputs/chunk_ltr_5000/chunk_eval_rankings_best_ensemble.csv`
- `outputs/chunk_ltr_5000/chunk_eval_rankings_best_ensemble.jsonl`
