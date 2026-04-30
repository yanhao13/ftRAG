#!/usr/bin/env python3
"""Merge a LoRA adapter into the base sequence-classification model.

Mirrors how data/lora_reranker.py loads the base model
(AutoModelForSequenceClassification with num_labels=1, ignore_mismatched_sizes=True),
attaches the trained adapter, calls PEFT's merge_and_unload() to fold the LoRA
weights into the base tensors, and writes a unified Hugging Face checkpoint that
downstream serving stacks can load directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown --dtype {name!r}; pick one of {sorted(mapping)}")
    return mapping[name]


def merge(
    base_model: str,
    adapter_path: Path,
    merged_output: Path,
    dtype: torch.dtype,
) -> None:
    print(f"loading base model: {base_model} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    print(f"loading adapter: {adapter_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json missing under {adapter_path}; "
            "did the training Job finish writing the adapter directory?"
        )

    peft_model = PeftModel.from_pretrained(base, str(adapter_path))

    print("calling merge_and_unload()")
    merged = peft_model.merge_and_unload()

    merged_output.mkdir(parents=True, exist_ok=True)
    print(f"saving merged model to {merged_output}")
    merged.save_pretrained(str(merged_output), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_output))

    manifest = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "dtype": str(dtype).replace("torch.", ""),
        "merged_output": str(merged_output),
    }
    (merged_output / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("done")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Hugging Face base model id; meta-llama/Llama-3.2-1B-Instruct also works if gated access is set up.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        type=Path,
        help="Local path to the saved PEFT adapter directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--merged-output",
        required=True,
        type=Path,
        help="Local path where the merged HF checkpoint will be written.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Torch dtype for the merged weights; float16, bfloat16, or float32.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    merge(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_output=args.merged_output,
        dtype=_torch_dtype(args.dtype),
    )


if __name__ == "__main__":
    main()
