#!/usr/bin/env python3
"""LoRA fine-tuning for the document/chunk ranking JSONL files.

The script treats each ranking prompt as a retrieve-and-rerank problem:
score every (question, candidate) pair with a Llama sequence-classification
reranker, then sort the candidates by score. This is much faster than SFT over
the full prompt, especially for the large chunk-ranking rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


QUESTION_RE = re.compile(
    r"Question:\s*(.*?)(?:\nText chunks:|\n\nDocument Types to rank:)",
    re.DOTALL,
)
ITEM_RE = re.compile(r"\[(Document|Chunk) Index (\d+)\]\s*")


@dataclass
class RankingQuery:
    qid: str
    task: str
    question: str
    candidates: list[tuple[int, str]]
    qrel: dict[int, float] | None


@dataclass
class PairExample:
    question: str
    candidate: str
    label: float


class PairDataset(Dataset):
    def __init__(self, examples: list[PairExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairExample:
        return self.examples[index]


def parse_prompt(obj: dict, candidate_char_limit: int) -> RankingQuery:
    prompt = obj["messages"][0]["content"]
    question_match = QUESTION_RE.search(prompt)
    if not question_match:
        raise ValueError("Could not find question in prompt")

    question = " ".join(question_match.group(1).split())
    matches = list(ITEM_RE.finditer(prompt))
    if not matches:
        raise ValueError("Could not find ranking candidates in prompt")

    task = "document" if matches[0].group(1) == "Document" else "chunk"
    candidates: list[tuple[int, str]] = []
    for pos, match in enumerate(matches):
        start = match.end()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(prompt)
        candidate = prompt[start:end].strip()
        if candidate_char_limit > 0:
            candidate = candidate[:candidate_char_limit]
        candidates.append((int(match.group(2)), candidate))

    qrel = None
    if "qrel" in obj:
        qrel = {int(k): float(v) for k, v in obj["qrel"].items()}

    return RankingQuery(
        qid=str(obj.get("uuid") or obj.get("_id")),
        task=task,
        question=question,
        candidates=candidates,
        qrel=qrel,
    )


def load_queries(
    path: Path,
    candidate_char_limit: int,
    limit: int,
    require_qrels: bool,
) -> list[RankingQuery]:
    queries: list[RankingQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"read {path.name}"):
            obj = json.loads(line)
            if require_qrels and "qrel" not in obj:
                continue
            queries.append(parse_prompt(obj, candidate_char_limit))
            if limit > 0 and len(queries) >= limit:
                break
    return queries


def split_queries(
    queries: list[RankingQuery],
    val_queries: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[RankingQuery], list[RankingQuery]]:
    if not queries:
        return [], []
    rng = random.Random(seed)
    shuffled = queries[:]
    rng.shuffle(shuffled)
    n_val = val_queries if val_queries > 0 else max(1, int(len(shuffled) * val_fraction))
    n_val = min(n_val, max(1, len(shuffled) - 1))
    return shuffled[n_val:], shuffled[:n_val]


def build_pair_examples(
    queries: Iterable[RankingQuery],
    negatives_per_positive: int,
    seed: int,
) -> list[PairExample]:
    rng = random.Random(seed)
    examples: list[PairExample] = []

    for query in queries:
        if query.qrel is None:
            continue
        max_rel = max(query.qrel.values(), default=1.0) or 1.0
        positives = [idx for idx, _ in query.candidates if query.qrel.get(idx, 0.0) > 0.0]
        negatives = [idx for idx, _ in query.candidates if query.qrel.get(idx, 0.0) <= 0.0]
        candidate_by_idx = dict(query.candidates)

        if query.task == "document":
            selected = [idx for idx, _ in query.candidates]
        else:
            neg_count = min(len(negatives), max(1, negatives_per_positive) * max(1, len(positives)))
            selected = positives + rng.sample(negatives, neg_count)

        for idx in selected:
            rel = query.qrel.get(idx, 0.0)
            examples.append(
                PairExample(
                    question=query.question,
                    candidate=candidate_by_idx[idx],
                    label=float(rel / max_rel),
                )
            )

    rng.shuffle(examples)
    return examples


def make_collate_fn(tokenizer, max_length: int, device: torch.device):
    def collate(batch: list[PairExample]) -> dict[str, torch.Tensor]:
        features = tokenizer(
            [x.question for x in batch],
            [x.candidate for x in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        features = {key: value.to(device) for key, value in features.items()}
        features["labels"] = torch.tensor([x.label for x in batch], dtype=torch.float32, device=device)
        return features

    return collate


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def infer_lora_target_modules(model, requested: str) -> list[str]:
    if requested != "auto":
        return [x.strip() for x in requested.split(",") if x.strip()]

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if {"q_proj", "v_proj"}.issubset(module_names):
        return ["q_proj", "v_proj"]
    if {"query", "value"}.issubset(module_names):
        return ["query", "value"]
    raise ValueError(
        "Could not infer LoRA target modules. Pass --target-modules explicitly, "
        "for example --target-modules q_proj,v_proj."
    )


def infer_modules_to_save(model) -> list[str] | None:
    if hasattr(model, "score"):
        return ["score"]
    if hasattr(model, "classifier"):
        return ["classifier"]
    return None


def attach_lora(model, args):
    target_modules = infer_lora_target_modules(model, args.target_modules)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=infer_modules_to_save(model),
    )
    print(f"LoRA target modules: {target_modules}")
    return get_peft_model(model, config)


def logits_for_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch.pop("labels", None)
    outputs = model(**batch)
    if labels is not None:
        batch["labels"] = labels
    return outputs.logits.view(-1)


def train(model, tokenizer, train_examples: list[PairExample], args, device: torch.device):
    if args.epochs <= 0:
        return

    loader = DataLoader(
        PairDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length, device),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels")
            logits = model(**batch).logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            if step % args.log_every == 0:
                progress.set_postfix(loss=f"{running / args.log_every:.4f}")
                running = 0.0

        if len(loader) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1


@torch.no_grad()
def score_queries(
    model,
    tokenizer,
    queries: list[RankingQuery],
    args,
    device: torch.device,
) -> dict[str, list[tuple[int, float]]]:
    model.eval()
    scored: dict[str, list[tuple[int, float]]] = {}

    for query in tqdm(queries, desc="score queries"):
        pairs = [(query.question, text) for _, text in query.candidates]
        scores: list[float] = []
        for start in range(0, len(pairs), args.eval_batch_size):
            chunk = pairs[start : start + args.eval_batch_size]
            features = tokenizer(
                [x[0] for x in chunk],
                [x[1] for x in chunk],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            features = {key: value.to(device) for key, value in features.items()}
            logits = model(**features).logits.view(-1)
            scores.extend(float(x) for x in logits.detach().cpu())

        ranked = sorted(
            zip([idx for idx, _ in query.candidates], scores),
            key=lambda item: (-item[1], item[0]),
        )
        scored[query.qid] = ranked

    return scored


def dcg(relevances: list[float], k: int) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def metrics_at_k(
    queries: list[RankingQuery],
    rankings: dict[str, list[tuple[int, float]]],
    k: int,
) -> dict[str, float]:
    ndcg_values: list[float] = []
    ap_values: list[float] = []
    rr_values: list[float] = []

    for query in queries:
        if query.qrel is None:
            continue
        ranked_ids = [idx for idx, _ in rankings[query.qid]]
        rel_by_idx = query.qrel
        ranked_rels = [rel_by_idx.get(idx, 0.0) for idx in ranked_ids]
        ideal_rels = sorted(rel_by_idx.values(), reverse=True)
        ideal = dcg(ideal_rels, k)
        ndcg_values.append(dcg(ranked_rels, k) / ideal if ideal > 0 else 0.0)

        num_relevant = sum(1 for rel in rel_by_idx.values() if rel > 0)
        denom = min(num_relevant, k)
        hits = 0
        precision_sum = 0.0
        reciprocal_rank = 0.0
        for rank, idx in enumerate(ranked_ids[:k], start=1):
            if rel_by_idx.get(idx, 0.0) > 0:
                hits += 1
                precision_sum += hits / rank
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
        ap_values.append(precision_sum / denom if denom else 0.0)
        rr_values.append(reciprocal_rank)

    count = len(ndcg_values)
    if count == 0:
        return {"queries": 0, f"nDCG@{k}": 0.0, f"MAP@{k}": 0.0, f"MRR@{k}": 0.0}
    return {
        "queries": count,
        f"nDCG@{k}": sum(ndcg_values) / count,
        f"MAP@{k}": sum(ap_values) / count,
        f"MRR@{k}": sum(rr_values) / count,
    }


def evaluate_by_task(
    model,
    tokenizer,
    queries: list[RankingQuery],
    args,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    rankings = score_queries(model, tokenizer, queries, args, device)
    results: dict[str, dict[str, float]] = {}
    for task in ["document", "chunk"]:
        task_queries = [q for q in queries if q.task == task]
        if task_queries:
            results[task] = metrics_at_k(task_queries, rankings, args.metric_k)
    return results


def write_rankings(
    path_prefix: Path,
    queries: list[RankingQuery],
    rankings: dict[str, list[tuple[int, float]]],
    chunk_output_k: int,
) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = path_prefix.with_suffix(".jsonl")
    csv_path = path_prefix.with_suffix(".csv")

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=["id", "task", "ranking"],
            lineterminator="\n",
        )
        writer.writeheader()
        for query in queries:
            ranked_ids = [idx for idx, _ in rankings[query.qid]]
            if query.task == "chunk":
                ranked_ids = ranked_ids[:chunk_output_k]
            row = {"id": query.qid, "task": query.task, "ranking": json.dumps(ranked_ids)}
            jsonl_handle.write(json.dumps(row) + "\n")
            writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", default="outputs/llama3_lora_reranker")
    parser.add_argument("--doc-dev", default="document_ranking_kaggle_dev.jsonl")
    parser.add_argument("--chunk-dev", default="chunk_ranking_kaggle_dev.jsonl")
    parser.add_argument("--doc-eval", default="document_ranking_kaggle_eval.jsonl")
    parser.add_argument("--chunk-eval", default="chunk_ranking_kaggle_eval.jsonl")
    parser.add_argument("--doc-query-limit", type=int, default=200, help="0 means all document dev rows; negative skips")
    parser.add_argument("--chunk-query-limit", type=int, default=20, help="0 means all chunk dev rows; negative skips")
    parser.add_argument("--doc-val-queries", type=int, default=40)
    parser.add_argument("--chunk-val-queries", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--candidate-char-limit", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-train-pairs", type=int, default=1000, help="0 means use all sampled pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="auto")
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--chunk-output-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--predict-eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"device: {device}")
    tokenizer, model = load_base_model(args.model_name)

    doc_queries = (
        []
        if args.doc_query_limit < 0
        else load_queries(Path(args.doc_dev), args.candidate_char_limit, args.doc_query_limit, True)
    )
    chunk_queries = (
        []
        if args.chunk_query_limit < 0
        else load_queries(Path(args.chunk_dev), args.candidate_char_limit, args.chunk_query_limit, True)
    )
    doc_train, doc_val = split_queries(doc_queries, args.doc_val_queries, args.val_fraction, args.seed)
    chunk_train, chunk_val = split_queries(chunk_queries, args.chunk_val_queries, args.val_fraction, args.seed + 1)
    train_queries = doc_train + chunk_train
    val_queries = doc_val + chunk_val
    train_examples = build_pair_examples(train_queries, args.negatives_per_positive, args.seed)
    if args.max_train_pairs > 0 and len(train_examples) > args.max_train_pairs:
        rng = random.Random(args.seed)
        rng.shuffle(train_examples)
        train_examples = train_examples[: args.max_train_pairs]

    print(f"train queries: {len(train_queries):,}")
    print(f"validation queries: {len(val_queries):,}")
    print(f"pair examples: {len(train_examples):,}")

    metrics: dict = {
        "config": vars(args),
        "data": {
            "train_queries": len(train_queries),
            "validation_queries": len(val_queries),
            "pair_examples": len(train_examples),
        },
    }

    model.to(device)
    if not args.skip_base_eval:
        metrics["base"] = evaluate_by_task(model, tokenizer, val_queries, args, device)
        save_json(output_dir / "metrics.json", metrics)

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        model = attach_lora(model, args)
    model.to(device)
    model.print_trainable_parameters()

    if not args.skip_train:
        train(model, tokenizer, train_examples, args, device)
        model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")

    metrics["fine_tuned"] = evaluate_by_task(model, tokenizer, val_queries, args, device)
    save_json(output_dir / "metrics.json", metrics)

    if args.predict_eval:
        eval_queries = []
        eval_queries.extend(load_queries(Path(args.doc_eval), args.candidate_char_limit, 0, False))
        eval_queries.extend(load_queries(Path(args.chunk_eval), args.candidate_char_limit, 0, False))
        eval_rankings = score_queries(model, tokenizer, eval_queries, args, device)
        write_rankings(output_dir / "eval_rankings", eval_queries, eval_rankings, args.chunk_output_k)

    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
