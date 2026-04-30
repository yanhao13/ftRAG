FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

ENV HF_HUB_DISABLE_XET=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY cloud/requirements-cloud.txt /tmp/requirements-cloud.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cloud.txt

ARG BASE_MODEL=unsloth/Llama-3.2-1B-Instruct
RUN python3 -c "import os; \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
name=os.environ.get('BAKE_MODEL','${BASE_MODEL}'); \
AutoTokenizer.from_pretrained(name); \
AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True); \
print('baked', name, 'into', os.environ['HF_HOME'])"

COPY data/lora_reranker.py /app/lora_reranker.py
COPY data/chunk_ranker.py /app/chunk_ranker.py
COPY data/chunk_best_ensemble.py /app/chunk_best_ensemble.py
COPY cloud/merge_lora_offline.py /app/merge_lora_offline.py

CMD ["python3", "/app/lora_reranker.py", "--help"]
