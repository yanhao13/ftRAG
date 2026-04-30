# FinRAG: Graph-Enhanced RAG with Fine-Tuning

FinRAG is an Open WebUI demo for financial filing question answering over dense, interdependent SEC-style disclosures.

Authors: Gaolin Qian (gq2142), Yanhao Bai (yb2630), Liang Song (ts3479)

## Demo Files

| File | Purpose |
| --- | --- |
| `finrag_demo.py` | Main Open WebUI Function / Pipe. Exposes the selectable model name `finrag` and uses FinRAG-specific planning, project context, filing search, and web search. |
| `granite_autogen_rag.py` | Original generic Granite AG2 retrieval agent, kept as a baseline/reference. |
| `image_researcher_granite_crewai.py` | Original Granite/CrewAI image researcher, kept as a separate demo. |

## What FinRAG Demonstrates

Financial filings are hard for standard RAG because evidence is scattered across narrative disclosures, risk factors, MD&A, statements, tables, and footnotes. A useful answer often has to connect a chain like:

```text
risk disclosure -> business / MD&A driver -> financial statement or P&L impact -> cited evidence span
```

FinRAG demonstrates that workflow with three retrieval channels:

- Built-in project context for explaining the FinRAG method, metrics, deployment, and future work.
- Open WebUI Knowledge search for user-uploaded SEC filings or graph-export chunks.
- Open WebUI web search for external public references.

## Method Summary

### Fine-Tuned Reranking

FinRAG uses a parameter-efficient LoRA reranker for chunk retrieval. The base Llama-3.2-1B-Instruct sequence classifier is frozen, small low-rank adapter matrices are trained, and each question-candidate pair receives a scalar relevance score. Candidates are sorted by score for top-k retrieval.

Reported ranking results:

| Metric | RFT Target | FinRAG | Delta |
| --- | ---: | ---: | ---: |
| nDCG@5 | 0.371 | 0.432 | +0.061 |
| MAP@5 | 0.274 | 0.391 | +0.117 |
| MRR@5 | 0.587 | 0.640 | +0.053 |

The local LoRA smoke run validated the adapter path and scoring loop on 120 pair examples and 26 train queries.

### Graph-Enhanced Retrieval

FinRAG combines vanilla hybrid retrieval with graph expansion:

- BM25 + dense retrieval.
- Structural graph: `Document -> Page -> Block`, where blocks can be sections, paragraphs, or tables.
- Semantic overlay: `Block -> CausalClaim -> Risk Entity / P&L Driver`.
- Expansion over same-page blocks, nearby blocks, section context, and claim-linked evidence.

Knowledge graph scale:

- 84 filing PDFs.
- 12,041 pages.
- 118,160 blocks: 45,987 paragraphs, 42,449 sections, 29,724 tables.
- 4,655 causal claims.
- 2,651 risk entities.
- 1,275 P&L drivers.

## Setup

### 1. Install Ollama and Pull a Model

The demo defaults to Granite through Ollama:

```bash
ollama pull ibm/granite4:latest
```

You can change the underlying model in the Open WebUI function settings through `TASK_MODEL_ID`.

### 2. Install Open WebUI

```bash
pip install open-webui
open-webui serve
```

Open `http://localhost:8080`.

### 3. Import FinRAG into Open WebUI

1. Go to `Admin Panel -> Functions -> +`.
2. Name the function `finrag`.
3. Paste the contents of `finrag_demo.py`.
4. Save and enable the function.
5. Select `FinRAG: Graph-Enhanced RAG with Fine-Tuning` in the model picker.

The selectable model ID exposed by the pipe is `finrag`; the actual inference model remains configurable with `TASK_MODEL_ID`.

### 4. Optional: Load SEC Filings

To answer real filing questions, upload filings or graph-export chunks:

1. Go to `Workspace -> Knowledge`.
2. Create a collection.
3. Upload 10-K PDFs, extracted text, table chunks, or graph-expanded evidence chunks.
4. Ask FinRAG questions that require risk, MD&A, statement, or footnote evidence.

### 5. Optional: Configure Web Search

FinRAG uses Open WebUI's search API. Configure DuckDuckGo, SearXNG, or another provider in Open WebUI settings. If no provider is configured, the pipe falls back to the `WEB_SEARCH_ENGINE` valve default.

## Configuration

| Valve | Default | Meaning |
| --- | --- | --- |
| `TASK_MODEL_ID` | `ibm/granite4:latest` | Underlying Ollama model used for planning, retrieval decisions, and answer writing. |
| `OPENAI_API_URL` | `http://localhost:11434` | Ollama endpoint used by AG2. |
| `OPENAI_API_KEY` | `ollama` | Placeholder key for local Ollama. |
| `MODEL_TEMPERATURE` | `0` | Deterministic demo behavior. |
| `MAX_PLAN_STEPS` | `3` | Maximum retrieval steps per question. |
| `NUM_CTX` | `4096` | Ollama context window. Keep this modest for local machines. |
| `WEB_SEARCH_ENGINE` | `duckduckgo` | Search engine name passed to Open WebUI retrieval. |
| `WEB_SEARCH_RESULT_COUNT` | `3` | Number of web results requested. |
| `ENABLE_PROJECT_CONTEXT` | `True` | Enables the built-in FinRAG project-context tool. |

## Example Prompts

Project demo questions:

```text
What is FinRAG, and how does graph-enhanced retrieval improve financial filing QA?
```

```text
Summarize our LoRA reranker results and compare them to the reinforcement fine-tuning target.
```

```text
Explain the FinRAG deployment story and how it connects to cloud computing and deep neural networks.
```

Filing QA questions after uploading SEC filings:

```text
What risks could pressure gross margin, and what filing evidence links those risks to P&L impact?
```

```text
Find evidence that connects foreign currency risk to revenue, expenses, or net income.
```

```text
Answer using a risk -> driver -> financial impact chain, and cite the filing sections or pages you used.
```

## Evaluation Notes

- Dense + BM25 remained strongest on FinanceBench numeric extraction questions.
- Dense + graph was preferred by an LLM judge on about 70% of risk/P&L causality questions.
- The final ranking artifacts were `chunk_eval_rankings_best_ensemble.csv` and `chunk_eval_rankings_best_ensemble.jsonl`.

## Future Work

- Train on the full FinAgentBench set with longer 8K-character candidate context.
- Scale to Llama-3.2 8B on A100.
- Serve with vLLM, continuous batching, and INT8/AWQ quantization.
- Add cross-document entity resolution.
- Expand end-to-end QA evaluation.
- Move distributed training to Ray Train on KubeRay when GPU quota is available.
