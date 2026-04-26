#!/usr/bin/env python3
"""Materialize the strongest cached chunk-ranking ensemble.

This script uses the cached feature packs and trained rankers from
``chunk_ranker.py``. It is intentionally lightweight: no training or feature
extraction is done unless the caches are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

import chunk_ranker


DEFAULT_WEIGHTS = {
    "rank_gain011": 0.85,
    "extra": 0.15,
}

FEATURE_COLUMNS = {
    "bm25": 16,
    "word": 17,
    "char": 18,
    "bm25_norm": 19,
    "word_norm": 20,
    "char_norm": 21,
    "bm25_rr": 22,
    "word_rr": 23,
    "char_rr": 24,
    "blend": 25,
}


def parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in raw.split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def load_feature_pack(path: Path) -> chunk_ranker.FeaturePack:
    # Feature packs were written when chunk_ranker.py ran as __main__, so keep
    # that pickle name resolvable from this small materialization script.
    sys.modules["__main__"].FeaturePack = chunk_ranker.FeaturePack
    return joblib.load(path)


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def load_components(args: argparse.Namespace, weights: dict[str, float]) -> dict[str, object]:
    components: dict[str, object] = {}
    base_models = joblib.load(args.models_path)

    for name in weights:
        if name in FEATURE_COLUMNS:
            components[name] = FEATURE_COLUMNS[name]
        elif name in base_models:
            components[name] = base_models[name]
        elif name == "rank_bin_350":
            components[name] = joblib.load(args.rank_bin_350_path)
        elif name == "rank_bin_650":
            components[name] = joblib.load(args.rank_bin_650_path)
        elif name == "rank_gain011":
            components[name] = joblib.load(args.rank_gain011_path)
        elif name == "rank_gain012":
            components[name] = joblib.load(args.rank_gain012_path)
        else:
            raise ValueError(f"Unknown ensemble component: {name}")

    return components


def predict_rankings(
    pack: chunk_ranker.FeaturePack,
    components: dict[str, object],
    weights: dict[str, float],
) -> list[list[tuple[int, float]]]:
    raw_scores: dict[str, np.ndarray] = {}
    for name, component in components.items():
        if isinstance(component, int):
            raw_scores[name] = pack.x[:, component].astype(np.float32)
        else:
            raw_scores[name] = component.predict(pack.x).astype(np.float32)

    rankings: list[list[tuple[int, float]]] = []
    offset = 0
    for ids, group_size in zip(pack.candidate_ids, pack.groups):
        scores = np.zeros(group_size, dtype=np.float32)
        for name, raw in raw_scores.items():
            segment = normalized(raw[offset : offset + group_size])
            scores += weights[name] * segment
        rankings.append(sorted(zip(ids, map(float, scores)), key=lambda item: (-item[1], item[0])))
        offset += group_size
    return rankings


def dcg(relevances: list[float], k: int) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def row_metrics_at_k(
    queries: list,
    rankings: list[list[tuple[int, float]]],
    k: int,
) -> dict[str, float]:
    ndcg_values: list[float] = []
    ap_values: list[float] = []
    rr_values: list[float] = []

    for query, ranking in zip(queries, rankings):
        if query.qrel is None:
            continue
        ranked_ids = [idx for idx, _ in ranking]
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


def write_chunk_rankings_rows(
    path_prefix: Path,
    queries: list,
    rankings: list[list[tuple[int, float]]],
    top_k: int,
) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as jsonl_handle, path_prefix.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=["id", "ranking"])
        writer.writeheader()
        for query, ranking in zip(queries, rankings):
            row = {"id": query.qid, "ranking": json.dumps([idx for idx, _ in ranking][:top_k])}
            jsonl_handle.write(json.dumps(row) + "\n")
            writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chunk_ltr_5000"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/chunk_ltr_cache"))
    parser.add_argument("--val-cache", type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--models-path", type=Path)
    parser.add_argument("--rank-bin-350-path", type=Path)
    parser.add_argument("--rank-bin-650-path", type=Path)
    parser.add_argument("--rank-gain011-path", type=Path)
    parser.add_argument("--rank-gain012-path", type=Path)
    parser.add_argument("--weights", default=",".join(f"{name}={value}" for name, value in DEFAULT_WEIGHTS.items()))
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--output-k", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.val_cache = args.val_cache or args.cache_dir / "val_500_6000_13.joblib"
    args.eval_cache = args.eval_cache or args.cache_dir / "eval_200_6000.joblib"
    args.models_path = args.models_path or args.output_dir / "models.joblib"
    args.rank_bin_350_path = args.rank_bin_350_path or args.output_dir / "rank_bin_350.joblib"
    args.rank_bin_650_path = args.rank_bin_650_path or args.output_dir / "rank_bin_650.joblib"
    args.rank_gain011_path = args.rank_gain011_path or args.output_dir / "rank_gain011.joblib"
    args.rank_gain012_path = args.rank_gain012_path or args.output_dir / "rank_gain012.joblib"

    weights = parse_weights(args.weights)
    components = load_components(args, weights)

    val_pack = load_feature_pack(args.val_cache)
    val_rankings = predict_rankings(val_pack, components, weights)
    validation = row_metrics_at_k(val_pack.queries, val_rankings, args.metric_k)
    qid_counts = Counter(query.qid for query in val_pack.queries)
    duplicate_qids = sum(1 for count in qid_counts.values() if count > 1)

    metrics = {
        "weights": weights,
        "validation": validation,
        "validation_metric": "row_level_duplicate_safe",
        "validation_duplicate_qids": duplicate_qids,
        "table3_target": {"nDCG@5": 0.371, "MAP@5": 0.274, "MRR@5": 0.587},
        "table3_delta": {
            key: validation[key] - target
            for key, target in {"nDCG@5": 0.371, "MAP@5": 0.274, "MRR@5": 0.587}.items()
        },
        "artifacts": {
            "validation_rankings": str(args.output_dir / "validation_rankings_best_ensemble"),
        },
    }

    write_chunk_rankings_rows(args.output_dir / "validation_rankings_best_ensemble", val_pack.queries, val_rankings, args.output_k)

    if not args.skip_eval:
        eval_pack = load_feature_pack(args.eval_cache)
        eval_rankings = predict_rankings(eval_pack, components, weights)
        write_chunk_rankings_rows(args.output_dir / "chunk_eval_rankings_best_ensemble", eval_pack.queries, eval_rankings, args.output_k)
        metrics["artifacts"]["chunk_eval_rankings"] = str(args.output_dir / "chunk_eval_rankings_best_ensemble")

    save_json(args.output_dir / "metrics_best_ensemble.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
