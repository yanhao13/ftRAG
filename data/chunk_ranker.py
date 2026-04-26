#!/usr/bin/env python3
"""Train and run a supervised chunk-ranking model.

This is a fast learning-to-rank companion for the Llama LoRA reranker. It uses
the labeled chunk dev JSONL to learn top-5 ordering directly, then writes chunk
rankings for the unlabeled eval JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRanker
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm.auto import tqdm

from lora_reranker import RankingQuery, metrics_at_k, parse_prompt, write_rankings


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?%?")
QUESTION_ENTITY_RE = re.compile(r"Question:\s*(?:How has|What|Which|Why|When|Where|Who|Did|Does|Do|Is|Are|Was|Were)?\s*([^?,'’]+?)(?:'s|’s|\s+(?:announced|reported|offered|provided|indicated|disclosed|described|discussed|outlined|changed|trended|compare|compared|affect|impact|mean|show|suggest)|\?)", re.I)

STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "question",
    "answer",
    "chunk",
    "index",
    "text",
    "document",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "technologies",
    "holdings",
    "plc",
    "ltd",
}


@dataclass
class FeaturePack:
    queries: list[RankingQuery]
    x: np.ndarray
    y: np.ndarray | None
    groups: list[int]
    candidate_ids: list[list[int]]


def load_chunk_queries(path: Path, limit: int, candidate_char_limit: int, require_qrels: bool) -> list[RankingQuery]:
    queries: list[RankingQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"read {path.name}"):
            obj = json.loads(line)
            if require_qrels and "qrel" not in obj:
                continue
            queries.append(parse_prompt(obj, candidate_char_limit=candidate_char_limit))
            if limit > 0 and len(queries) >= limit:
                break
    return queries


def split_queries(
    queries: list[RankingQuery],
    val_queries: int,
    seed: int,
) -> tuple[list[RankingQuery], list[RankingQuery]]:
    rng = random.Random(seed)
    shuffled = queries[:]
    rng.shuffle(shuffled)
    val_queries = min(max(1, val_queries), max(1, len(shuffled) - 1))
    return shuffled[val_queries:], shuffled[:val_queries]


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    found = [tok.lower() for tok in TOKEN_RE.findall(text)]
    if keep_stopwords:
        return found
    return [tok for tok in found if tok not in STOP_WORDS and len(tok) > 1]


def bigrams(items: list[str]) -> set[tuple[str, str]]:
    return set(zip(items, items[1:]))


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def query_entity(question: str) -> str:
    match = QUESTION_ENTITY_RE.search(f"Question: {question}")
    if not match:
        return ""
    return " ".join(tokens(match.group(1), keep_stopwords=False))


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs_tokens:
        return []
    n_docs = len(docs_tokens)
    avgdl = sum(len(doc) for doc in docs_tokens) / max(n_docs, 1)
    df: Counter[str] = Counter()
    for doc in docs_tokens:
        for tok in set(doc):
            df[tok] += 1

    qtf = Counter(query_tokens)
    scores: list[float] = []
    for doc in docs_tokens:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for tok, q_count in qtf.items():
            if tok not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df[tok] + 0.5) / (df[tok] + 0.5))
            denom = tf[tok] + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
            score += q_count * idf * (tf[tok] * (k1 + 1.0) / denom)
        scores.append(score)
    return scores


def safe_tfidf_scores(question: str, docs: list[str], **kwargs) -> list[float]:
    try:
        matrix = TfidfVectorizer(**kwargs).fit_transform([question] + docs)
        return linear_kernel(matrix[0:1], matrix[1:]).ravel().astype(float).tolist()
    except ValueError:
        return [0.0] * len(docs)


def reciprocal_ranks(scores: list[float], ids: list[int]) -> list[float]:
    order = sorted(range(len(scores)), key=lambda pos: (-scores[pos], ids[pos]))
    ranks = [0] * len(scores)
    for rank, pos in enumerate(order, start=1):
        ranks[pos] = rank
    return [1.0 / rank for rank in ranks]


def normalize_query_scores(scores: list[float]) -> list[float]:
    arr = np.asarray(scores, dtype=np.float32)
    if len(arr) == 0:
        return []
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return [0.0] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()


def query_features(query: RankingQuery) -> tuple[list[int], list[list[float]]]:
    ids = [idx for idx, _ in query.candidates]
    docs = [text for _, text in query.candidates]
    clipped_docs = [text[:6000] for text in docs]

    q_tokens = tokens(query.question)
    q_tokens_all = tokens(query.question, keep_stopwords=True)
    q_set = set(q_tokens)
    q_bigram = bigrams(q_tokens)
    q_numbers = numbers(query.question)
    entity = query_entity(query.question)
    entity_tokens = set(tokens(entity))

    doc_tokens = [tokens(text) for text in clipped_docs]
    doc_tokens_all = [tokens(text, keep_stopwords=True) for text in clipped_docs]
    bm25 = bm25_scores(q_tokens, doc_tokens)
    word_tfidf = safe_tfidf_scores(
        query.question,
        clipped_docs,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    char_tfidf = safe_tfidf_scores(
        query.question,
        clipped_docs,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        norm="l2",
    )
    word_rr = reciprocal_ranks(word_tfidf, ids)
    char_rr = reciprocal_ranks(char_tfidf, ids)
    bm25_rr = reciprocal_ranks(bm25, ids)
    word_norm = normalize_query_scores(word_tfidf)
    char_norm = normalize_query_scores(char_tfidf)
    bm25_norm = normalize_query_scores(bm25)

    features: list[list[float]] = []
    n_candidates = len(ids)
    for pos, (idx, text) in enumerate(query.candidates):
        lower = text.lower()
        prefix = lower[:1200]
        heading = lower.split("\n", 1)[0][:300]
        doc_set = set(doc_tokens[pos])
        doc_bigram = bigrams(doc_tokens[pos])
        doc_numbers = numbers(text[:6000])
        overlap = len(q_set & doc_set)
        overlap_all = len(set(q_tokens_all) & set(doc_tokens_all[pos]))
        bigram_overlap = len(q_bigram & doc_bigram)
        entity_overlap = len(entity_tokens & doc_set)
        score_blend = bm25_norm[pos] + word_norm[pos] + char_norm[pos]

        features.append(
            [
                float(idx),
                float(idx / max(n_candidates - 1, 1)),
                float(n_candidates),
                math.log1p(len(text)),
                math.log1p(min(len(text), 6000)),
                math.log1p(len(doc_tokens[pos])),
                float(overlap),
                float(overlap / max(len(q_set), 1)),
                float(overlap / max(len(doc_set), 1)),
                float(overlap_all / max(len(set(q_tokens_all)), 1)),
                float(bigram_overlap),
                float(bigram_overlap / max(len(q_bigram), 1)),
                float(entity_overlap),
                float(entity_overlap / max(len(entity_tokens), 1)),
                float(len(q_numbers & doc_numbers)),
                float(len(q_numbers & doc_numbers) / max(len(q_numbers), 1)),
                float(bm25[pos]),
                float(word_tfidf[pos]),
                float(char_tfidf[pos]),
                float(bm25_norm[pos]),
                float(word_norm[pos]),
                float(char_norm[pos]),
                float(bm25_rr[pos]),
                float(word_rr[pos]),
                float(char_rr[pos]),
                float(score_blend),
                float(1.0 / (pos + 1)),
                float("#" in text[:100]),
                float("<table>" in prefix),
                float("table of contents" in prefix),
                float("annual report" in prefix or "form 10-k" in prefix),
                float("quarterly report" in prefix or "form 10-q" in prefix),
                float("earnings" in prefix or "conference call" in prefix),
                float(any(term in prefix for term in ["guidance", "outlook", "forecast", "target"])),
                float(any(term in prefix for term in ["revenue", "sales", "net sales"])),
                float(any(term in prefix for term in ["margin", "profit", "profitability", "income"])),
                float(any(term in prefix for term in ["cash", "debt", "liquidity", "capital"])),
                float(any(term in prefix for term in ["stock", "share", "ownership", "equity"])),
                float(any(term in prefix for term in ["segment", "business unit", "division"])),
                float(any(term in prefix for term in ["risk", "uncertain", "headwind"])),
                float(any(tok in heading for tok in q_set)),
            ]
        )
    return ids, features


def featurize(queries: list[RankingQuery], include_labels: bool) -> FeaturePack:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    groups: list[int] = []
    candidate_ids: list[list[int]] = []

    for query in tqdm(queries, desc="extract features"):
        ids, features = query_features(query)
        x_rows.extend(features)
        groups.append(len(features))
        candidate_ids.append(ids)
        if include_labels:
            assert query.qrel is not None
            y_rows.extend([query.qrel.get(idx, 0.0) for idx in ids])

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32) if include_labels else None
    return FeaturePack(queries=queries, x=x, y=y, groups=groups, candidate_ids=candidate_ids)


def load_or_build_cache(path: Path, queries: list[RankingQuery], include_labels: bool, refresh: bool) -> FeaturePack:
    if path.exists() and not refresh:
        return joblib.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack = featurize(queries, include_labels)
    joblib.dump(pack, path)
    return pack


def fit_models(train: FeaturePack, seed: int):
    assert train.y is not None
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=47,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.88,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.15,
        label_gain=[0, 1, 4],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(train.x, train.y.astype(int), group=train.groups)

    extra = ExtraTreesRegressor(
        n_estimators=180,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    extra.fit(train.x, train.y)

    hgb = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=seed,
    )
    hgb.fit(train.x, train.y)
    return {"lgbm": ranker, "extra": extra, "hgb": hgb}


def fit_aux_rankers(train: FeaturePack) -> dict[str, LGBMRanker]:
    """Train the auxiliary LightGBM rankers used by the best chunk ensemble."""
    assert train.y is not None
    binary_y = (train.y > 0).astype(int)
    graded_y = train.y.astype(int)

    specs = {
        "rank_bin_350": {
            "y": binary_y,
            "n_estimators": 350,
            "learning_rate": 0.045,
            "label_gain": [0, 1],
            "random_state": 21,
        },
        "rank_bin_650": {
            "y": binary_y,
            "n_estimators": 650,
            "learning_rate": 0.03,
            "label_gain": [0, 1],
            "random_state": 22,
        },
        "rank_gain011": {
            "y": graded_y,
            "n_estimators": 550,
            "learning_rate": 0.035,
            "label_gain": [0, 1, 1],
            "random_state": 23,
        },
        "rank_gain012": {
            "y": graded_y,
            "n_estimators": 550,
            "learning_rate": 0.035,
            "label_gain": [0, 1, 2],
            "random_state": 24,
        },
    }

    rankers: dict[str, LGBMRanker] = {}
    for name, spec in specs.items():
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=spec["n_estimators"],
            learning_rate=spec["learning_rate"],
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20 if name.startswith("rank_gain") else 18,
            subsample=0.9,
            colsample_bytree=0.88 if name.startswith("rank_gain") else 0.9,
            reg_alpha=0.02,
            reg_lambda=0.1,
            label_gain=spec["label_gain"],
            random_state=spec["random_state"],
            n_jobs=-1,
            verbose=-1,
        )
        ranker.fit(train.x, spec["y"], group=train.groups)
        rankers[name] = ranker
    return rankers


def grouped_predictions(pack: FeaturePack, models: dict, weights: dict[str, float]) -> dict[str, list[tuple[int, float]]]:
    raw_scores = {name: model.predict(pack.x).astype(np.float32) for name, model in models.items()}
    rankings: dict[str, list[tuple[int, float]]] = {}
    offset = 0
    for query, ids, group_size in zip(pack.queries, pack.candidate_ids, pack.groups):
        score = np.zeros(group_size, dtype=np.float32)
        for name, values in raw_scores.items():
            weight = weights.get(name, 0.0)
            if weight == 0.0:
                continue
            chunk = values[offset : offset + group_size]
            lo = float(chunk.min())
            hi = float(chunk.max())
            if hi > lo:
                chunk = (chunk - lo) / (hi - lo)
            score += weight * chunk
        rankings[query.qid] = sorted(zip(ids, map(float, score)), key=lambda item: (-item[1], item[0]))
        offset += group_size
    return rankings


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_chunk_rankings(path_prefix: Path, queries: list[RankingQuery], rankings: dict[str, list[tuple[int, float]]], top_k: int) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as jsonl_handle, path_prefix.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=["id", "ranking"],
            lineterminator="\n",
        )
        writer.writeheader()
        for query in queries:
            ranking = [idx for idx, _ in rankings[query.qid]][:top_k]
            row = {"id": query.qid, "ranking": json.dumps(ranking)}
            jsonl_handle.write(json.dumps(row) + "\n")
            writer.writerow(row)


def parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in raw.split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dev", default="chunk_ranking_kaggle_dev.jsonl")
    parser.add_argument("--chunk-eval", default="chunk_ranking_kaggle_eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/chunk_ltr")
    parser.add_argument("--cache-dir", default="outputs/chunk_ltr/cache")
    parser.add_argument("--train-query-limit", type=int, default=5000)
    parser.add_argument("--val-queries", type=int, default=500)
    parser.add_argument("--candidate-char-limit", type=int, default=6000)
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--output-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--weights", default="lgbm=0.55,extra=0.35,hgb=0.10")
    parser.add_argument("--train-aux-rankers", action="store_true")
    parser.add_argument("--predict-eval", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_limit = args.train_query_limit + args.val_queries if args.train_query_limit > 0 else 0
    queries = load_chunk_queries(Path(args.chunk_dev), total_limit, args.candidate_char_limit, require_qrels=True)
    train_queries, val_queries = split_queries(queries, args.val_queries, args.seed)
    if args.train_query_limit > 0:
        train_queries = train_queries[: args.train_query_limit]
    print(f"train chunk queries: {len(train_queries):,}")
    print(f"validation chunk queries: {len(val_queries):,}")

    train_pack = load_or_build_cache(cache_dir / f"train_{len(train_queries)}_{args.candidate_char_limit}_{args.seed}.joblib", train_queries, True, args.refresh_cache)
    val_pack = load_or_build_cache(cache_dir / f"val_{len(val_queries)}_{args.candidate_char_limit}_{args.seed}.joblib", val_queries, True, args.refresh_cache)
    models = fit_models(train_pack, args.seed)
    joblib.dump(models, out_dir / "models.joblib")
    if args.train_aux_rankers:
        for name, model in fit_aux_rankers(train_pack).items():
            joblib.dump(model, out_dir / f"{name}.joblib")

    weights = parse_weights(args.weights)
    val_rankings = grouped_predictions(val_pack, models, weights)
    metrics = {
        "config": vars(args),
        "weights": weights,
        "validation": metrics_at_k(val_queries, val_rankings, args.metric_k),
        "table3_target": {"nDCG@5": 0.371, "MAP@5": 0.274, "MRR@5": 0.587},
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_chunk_rankings(out_dir / "validation_rankings", val_queries, val_rankings, args.output_k)

    if args.predict_eval:
        eval_queries = load_chunk_queries(Path(args.chunk_eval), 0, args.candidate_char_limit, require_qrels=False)
        eval_pack = load_or_build_cache(cache_dir / f"eval_{len(eval_queries)}_{args.candidate_char_limit}.joblib", eval_queries, False, args.refresh_cache)
        eval_rankings = grouped_predictions(eval_pack, models, weights)
        write_chunk_rankings(out_dir / "chunk_eval_rankings", eval_queries, eval_rankings, args.output_k)

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
