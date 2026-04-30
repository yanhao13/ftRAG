"""
requirements: ag2==0.9.10, ag2[ollama]==0.9.10, ag2[openai]==0.9.10
"""

from fastapi import Request
from autogen import ConversableAgent
from typing import Annotated, Awaitable, Callable, Optional
from open_webui.routers import retrieval
from open_webui.models.knowledge import KnowledgeTable
from open_webui import config as open_webui_config
from pydantic import BaseModel, Field
import inspect
import json
import logging
import re


FINRAG_PROJECT_CONTEXT = """
## FinRAG identity
FinRAG is a course demo model named "finrag": Graph-Enhanced RAG with Fine-Tuning.
Authors: Gaolin Qian (gq2142), Yanhao Bai (yb2630), Liang Song (ts3479).
The demo is designed for financial filing question answering over SEC-style disclosures.

## Why financial filings are hard for RAG
Financial filings are dense, heterogeneous, and interdependent. A 10-K combines
narrative disclosures, risk factors, MD&A, structured financial statements,
semi-structured tables, forward-looking language, footnotes, and cross-section
dependencies. Many answers require linking risk disclosure to business drivers,
then to a financial statement or footnote impact. Example chain: supply-chain
disruption risk -> higher component or freight costs -> higher COGS -> pressure
on gross margin.

## Recent-work framing
FinQA is useful for local text/table numerical reasoning but weak as an
end-to-end RAG benchmark. FinanceBench is a useful filing QA benchmark over
long documents, mostly single-document. FinBen is broad across financial tasks
but usually does not require retrieval over a corpus. FinAgentBench focuses on
agentic document and chunk ranking over SEC/public disclosures, but is closer
to an oracle-labeled retrieval benchmark than a full RAG QA benchmark.

## Fine-tuning method
The fine-tuning component is a parameter-efficient LoRA reranker for chunk
retrieval. The base model is an open-source Llama-3.2-1B-Instruct sequence
classifier. Base weights stay frozen; small trainable low-rank matrices A and B
learn task-specific ranking behavior through an effective W + BA update. Each
training example is a question-candidate pair. The model emits a scalar
relevance score for each candidate chunk, then FinRAG sorts candidates by score.
Smoke-run settings used r=8, alpha=16, dropout=0.05, and short candidate windows
for speed.

## Graph RAG method
FinRAG uses multi-channel retrieval: BM25 plus dense search, structural graph
lookup, semantic overlay lookup, and graph expansion. The structural graph is
Document -> Page -> Block, where blocks are Section, Paragraph, or Table nodes.
The semantic overlay links source blocks to normalized causal claims, risk
entities, and P&L drivers. Graph expansion pulls same-page blocks, nearby
blocks, section context, and claim-linked evidence before reranking.

## Graph construction scale
The Neo4j knowledge graph contains 84 filing PDFs, 12,041 pages, and 118,160
blocks: 45,987 paragraphs, 42,449 sections, and 29,724 tables. The semantic
overlay includes 4,655 causal claims, 2,651 risk entities, and 1,275 P&L
drivers. Example semantic claim: "the negative impact from unfavorable foreign
currency exchange rates", direction decreases, confidence 0.8, P&L driver
"negative impact".

## Evaluation
The final chunk-ranking output exceeds the reinforcement fine-tuning target on
all reported top-k retrieval metrics: nDCG@5 0.432 versus 0.371 (+0.061),
MAP@5 0.391 versus 0.274 (+0.117), and MRR@5 0.640 versus 0.587 (+0.053).
Metrics are row-level and duplicate-safe on a 500-row validation set with 51
duplicate query IDs. Output artifacts include
chunk_eval_rankings_best_ensemble.csv and
chunk_eval_rankings_best_ensemble.jsonl.

## LoRA smoke run
A small local Llama-3 run validates the adapter path and scoring loop. On 120
pair examples and 26 training queries, the fine-tuned reranker improved nDCG@5
from 0.741 to 0.752 and MAP@5 from 0.689 to 0.712, while MRR@5 remained 0.800.
The smoke run validates the method, not maximum accuracy; full training needs
more rows and longer candidate context.

## End-to-end findings
Dense plus BM25 remained strongest on FinanceBench, where many questions are
numeric extraction tasks. Dense plus graph was preferred by an LLM judge on
about 70% of risk/P&L causality questions, where evidence connectivity matters.
The judged risk/P&L setup used 2 personas, 2 causality tasks, and 2 questions,
for 8 judged comparisons.

## Deployment and course relevance
The project used a hybrid pipeline across Colab T4, GKE Autopilot, and GCS as
artifact glue. Workloads ran under cloud quota constraints. The project connects
cloud computing with deep neural networks through GKE/Colab execution, LoRA
fine-tuning, GraphRAG retrieval over SEC filings, and retrieval metric
evaluation.

## Future work
Future work includes migrating training back to GKE when GPU quota is granted,
benchmarking Colab T4 versus GKE L4 timing, scaling to Llama-3.2 8B on A100,
training on the full FinAgentBench set with longer 8K-character candidate
context, serving with vLLM, continuous batching benchmarks, INT8 or AWQ
quantization, cross-document entity resolution, end-to-end QA evaluation, and
distributed training with Ray Train on KubeRay.
"""

FINRAG_METRIC_STATEMENT = (
    "nDCG@5 0.432 versus 0.371 (+0.061), "
    "MAP@5 0.391 versus 0.274 (+0.117), and "
    "MRR@5 0.640 versus 0.587 (+0.053)"
)


PLANNER_MESSAGE = """
You are FinRAG's retrieval planner for financial filing QA and project demos.
Create a short, executable retrieval plan. The helper can:
1. Read built-in FinRAG project/demo context.
2. Search the user's Open WebUI knowledge bases, which may contain SEC filings.
3. Search the web through Open WebUI search.

Planning policy:
- For questions about FinRAG itself, use built-in FinRAG project context first.
- If the user asks what FinRAG is, how the LoRA reranker works, how GraphRAG
  works, what the metrics are, or how the demo was deployed, return exactly one
  project-context step. Do not add filing or web steps for these project-demo
  questions unless the user explicitly asks for external evidence.
- For SEC filing questions, collect linked evidence: risk disclosure, MD&A or
  business driver, financial statement/table/footnote impact, and source spans.
- For risk-to-P&L questions, explicitly retrieve causal-chain evidence.
- For benchmark, metric, LoRA, graph, deployment, or course-relevance questions,
  retrieve the corresponding FinRAG project context section.
- Keep the plan to at most 3 steps. Do not include summarization; another agent
  will write the final answer.

Return only JSON matching:
{"steps": ["step one", "step two"]}
"""


ASSISTANT_PROMPT = """
You are FinRAG's evidence collector. Complete exactly one retrieval step.

Available tools:
- finrag_project_context: built-in project facts about the FinRAG demo.
- filing_knowledge_search: Open WebUI knowledge search over user-uploaded
  financial filings and related documents.
- web_search: Open WebUI web search for external public references.

Rules:
1. Use tools when evidence is needed. Use at most one tool call at a time.
2. For financial filing QA, prefer filing_knowledge_search and preserve source
   details such as document, page, section, table, or footnote if returned.
3. For project/demo questions, prefer finrag_project_context.
4. Ground the step output only in provided context or tool outputs.
5. If a tool returns an error or no evidence, say that directly and state the next best
   retrieval channel.
6. Preserve metrics exactly as returned by tools. Do not convert absolute
   metric deltas into percentages.

Output one of:
##ANSWER## <concise evidence summary with sources if available>
##TERMINATE##
"""


REPORT_WRITER_PROMPT = """
You are the FinRAG demo answer writer.

Write a concise, presentation-ready answer using only the gathered evidence.
For financial filing answers, use this shape when possible:
- Answer
- Evidence chain: risk -> driver -> financial/P&L impact
- Sources
- Caveat if evidence is incomplete

For project-demo answers, explain FinRAG as Graph-Enhanced RAG with Fine-Tuning
and cite the relevant metric, graph, method, or deployment facts from the
gathered context.

Do not invent source URLs, page numbers, metrics, or graph counts. If evidence
contains exact metric values or deltas, copy them exactly. Do not convert metric
deltas into percentages unless the evidence explicitly gives percentages. Ignore
failed tool outputs as factual evidence unless the user is asking about the run
failure itself. Tool outputs are authoritative; if a researcher summary conflicts
with a tool output, use the tool output. For FinRAG evaluation metrics, the exact
statement is: nDCG@5 0.432 versus 0.371 (+0.061), MAP@5 0.391 versus 0.274
(+0.117), and MRR@5 0.640 versus 0.587 (+0.053). Do not rewrite this as 6.1%,
11.7%, or 5.3%.
"""


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9@+._/-]*", text.lower()))


def _select_project_context(query: str, max_sections: int = 5) -> str:
    sections = []
    for section in FINRAG_PROJECT_CONTEXT.strip().split("\n## "):
        section = section.strip()
        if not section:
            continue
        if not section.startswith("## "):
            section = "## " + section
        sections.append(section)

    query_tokens = _tokenize(query)
    scored_sections = []
    for section in sections:
        section_tokens = _tokenize(section)
        score = len(query_tokens & section_tokens)
        scored_sections.append((score, section))

    selected = [
        section
        for score, section in sorted(scored_sections, key=lambda item: item[0], reverse=True)
        if score > 0
    ][:max_sections]

    if not selected:
        selected = sections[:max_sections]

    required_section_markers = []
    if query_tokens & {"metric", "metrics", "ndcg", "map", "mrr", "improvement", "improvements", "delta"}:
        required_section_markers.append("## Evaluation")
    if query_tokens & {"lora", "reranker", "reranking", "fine-tuning", "finetuning"}:
        required_section_markers.append("## Fine-tuning method")
    if query_tokens & {"graph", "graphrag", "retrieval", "structural", "semantic"}:
        required_section_markers.append("## Graph RAG method")

    for marker in required_section_markers:
        required_section = next((section for section in sections if section.startswith(marker)), None)
        if required_section and required_section not in selected:
            selected.append(required_section)

    return "\n\n".join(selected)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _should_use_project_context_only(query: str) -> bool:
    query_tokens = _tokenize(query)
    project_terms = {
        "finrag",
        "graph",
        "graphrag",
        "lora",
        "reranker",
        "reranking",
        "fine-tuning",
        "finetuning",
        "metrics",
        "ndcg",
        "map",
        "mrr",
        "evaluation",
        "deployment",
        "course",
        "future",
        "demo",
    }
    filing_terms = {
        "10-k",
        "10-q",
        "filing",
        "sec",
        "annual",
        "quarterly",
        "bestbuy",
        "best",
        "buy",
        "gross",
        "margin",
        "risk",
        "revenue",
        "income",
        "cash",
        "footnote",
        "table",
    }
    external_terms = {"web", "latest", "current", "search", "online", "url", "source"}

    asks_about_project = bool(query_tokens & project_terms)
    asks_for_external_or_filing = bool(query_tokens & (filing_terms | external_terms))
    return asks_about_project and not asks_for_external_or_filing


def _asks_for_finrag_metrics(query: str) -> bool:
    query_tokens = _tokenize(query)
    return bool(query_tokens & {"metric", "metrics", "ndcg", "map", "mrr", "improvement", "improvements", "delta"})


def _enforce_finrag_metric_guardrail(answer: str, query: str) -> str:
    if not _asks_for_finrag_metrics(query):
        return answer

    lines = answer.splitlines()
    for index, line in enumerate(lines):
        normalized_line = line.lower()
        has_all_metric_names = all(term in normalized_line for term in ("ndcg", "map", "mrr"))
        has_metric_conversion = any(
            token in normalized_line
            for token in ("%", "6.1", "11.7", "5.3", "0.12", "0.08")
        )
        if not has_all_metric_names or not has_metric_conversion:
            continue

        bullet_match = re.match(r"^(\s*-\s*(?:\*\*[^*]+\*\*:\s*)?)", line)
        prefix = bullet_match.group(1) if bullet_match else ""
        lines[index] = f"{prefix}{FINRAG_METRIC_STATEMENT}."

    guarded_answer = "\n".join(lines)
    exact_metric_tokens = {"0.432", "0.371", "+0.061", "0.391", "0.274", "+0.117", "0.640", "0.587", "+0.053"}
    has_exact_metrics = all(token in guarded_answer for token in exact_metric_tokens)
    if not has_exact_metrics:
        guarded_answer = f"{guarded_answer.rstrip()}\n\n{FINRAG_METRIC_STATEMENT}."
    return guarded_answer


class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="ibm/granite4:latest")
        OPENAI_API_URL: str = Field(default="http://localhost:11434")
        OPENAI_API_KEY: str = Field(default="ollama")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=3)
        NUM_CTX: int = Field(default=4096)
        WEB_SEARCH_ENGINE: str = Field(default="duckduckgo")
        WEB_SEARCH_RESULT_COUNT: int = Field(default=3)
        ENABLE_PROJECT_CONTEXT: bool = Field(default=True)

    def __init__(self):
        self.type = "pipe"
        self.id = "finrag"
        self.name = "FinRAG: Graph-Enhanced RAG with Fine-Tuning"
        self.valves = self.Valves()

    def get_provider_models(self):
        return [
            {
                "id": self.id,
                "name": self.name,
            }
        ]

    def is_open_webui_request(self, body):
        message = str(body[-1])
        prompt_templates = {
            "### Task",
            open_webui_config.DEFAULT_RAG_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE.replace("\n", "\\n"),
        }
        return any(template and template[:50] in message for template in prompt_templates)

    async def emit_event_safe(self, message: str):
        if not self.event_emitter:
            return
        try:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {"content": message + "\n"},
                }
            )
        except Exception as exc:
            logging.error(f"Error emitting event: {exc}")

    def _latest_user_text(self, body: dict) -> str:
        latest_content = body["messages"][-1]["content"]
        if isinstance(latest_content, str):
            return latest_content

        text = ""
        for item in latest_content:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text.strip()

    def _ensure_web_search_config(self):
        config = self.owui_request.app.state.config
        defaults = {
            "WEB_SEARCH_ENGINE": self.valves.WEB_SEARCH_ENGINE,
            "WEB_SEARCH_RESULT_COUNT": self.valves.WEB_SEARCH_RESULT_COUNT,
            "WEB_SEARCH_DOMAIN_FILTER_LIST": [],
            "WEB_SEARCH_CONCURRENT_REQUESTS": 1,
            "DDGS_BACKEND": "auto",
        }
        for key, value in defaults.items():
            if not hasattr(config, key) or getattr(config, key) in (None, ""):
                setattr(config, key, value)
        return config

    async def pipe(
        self,
        body,
        __user__: Optional[dict],
        __request__: Request,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        default_model = self.valves.TASK_MODEL_ID
        base_url = self.valves.OPENAI_API_URL
        model_temp = self.valves.MODEL_TEMPERATURE
        max_plan_steps = self.valves.MAX_PLAN_STEPS
        self.event_emitter = __event_emitter__
        self.owui_request = __request__
        self.user = __user__

        class RetrievalPlan(BaseModel):
            steps: list[str]

        base_llm_config = {
            "model": default_model,
            "client_host": base_url,
            "api_type": "ollama",
            "temperature": model_temp,
            "num_ctx": self.valves.NUM_CTX,
        }

        generic_assistant = ConversableAgent(
            name="FinRAG_Generic_Assistant",
            llm_config={**base_llm_config, "config_list": [{**base_llm_config}]},
            human_input_mode="NEVER",
        )

        if self.is_open_webui_request(body["messages"]):
            reply = generic_assistant.generate_reply(messages=[body["messages"][-1]])
            return reply

        planner = ConversableAgent(
            name="FinRAG_Planner",
            system_message=PLANNER_MESSAGE,
            llm_config={
                **base_llm_config,
                "config_list": [{**base_llm_config, "response_format": RetrievalPlan}],
            },
            human_input_mode="NEVER",
        )

        researcher = ConversableAgent(
            name="FinRAG_Researcher",
            system_message=ASSISTANT_PROMPT,
            llm_config={**base_llm_config, "config_list": [{**base_llm_config}]},
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "##ANSWER##" in msg.get("content", "")
            or "##TERMINATE##" in msg.get("content", "")
            or ("tool_calls" not in msg and msg.get("content", "") == ""),
        )

        report_generator = ConversableAgent(
            name="FinRAG_Report_Writer",
            system_message=REPORT_WRITER_PROMPT,
            llm_config={**base_llm_config, "config_list": [{**base_llm_config}]},
            human_input_mode="NEVER",
        )

        user_proxy = ConversableAgent(
            name="FinRAG_User",
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "##ANSWER##" in msg.get("content", "")
            or "##TERMINATE##" in msg.get("content", "")
            or ("tool_calls" not in msg and msg.get("content", "") == ""),
        )
        latest_content = self._latest_user_text(body)

        @researcher.register_for_llm(
            name="finrag_project_context",
            description="Retrieve built-in facts about the FinRAG model, GraphRAG architecture, LoRA reranker, metrics, deployment, course relevance, and future work.",
        )
        @user_proxy.register_for_execution(name="finrag_project_context")
        def finrag_project_context(
            search_instruction: Annotated[str, "Question or topic to retrieve from the FinRAG project context."]
        ) -> str:
            if not self.valves.ENABLE_PROJECT_CONTEXT:
                return "Built-in FinRAG project context is disabled."
            combined_instruction = f"{latest_content}\n{search_instruction}".strip()
            return _select_project_context(combined_instruction)

        @researcher.register_for_llm(
            name="filing_knowledge_search",
            description="Search user-uploaded financial filings, SEC reports, tables, footnotes, and filing-derived graph evidence in Open WebUI Knowledge.",
        )
        @user_proxy.register_for_execution(name="filing_knowledge_search")
        async def filing_knowledge_search(
            search_instruction: Annotated[str, "Financial filing retrieval query."]
        ) -> str:
            if not search_instruction:
                return "Please provide a filing search query."

            knowledge_item_list = await _maybe_await(KnowledgeTable().get_knowledge_bases())
            if len(knowledge_item_list) == 0:
                return "No Open WebUI knowledge bases are available. Upload SEC filings or graph-export chunks to Workspace > Knowledge."

            collection_names = [item.id for item in knowledge_item_list]
            collection_form = retrieval.QueryCollectionsForm(
                collection_names=collection_names,
                query=search_instruction,
            )
            response = await _maybe_await(
                retrieval.query_collection_handler(
                    request=self.owui_request,
                    form_data=collection_form,
                    user=self.user,
                )
            )
            return json.dumps(response, default=str)

        @researcher.register_for_llm(
            name="web_search",
            description="Search the web through Open WebUI when external public information is needed.",
        )
        @user_proxy.register_for_execution(name="web_search")
        async def web_search(
            query: Annotated[str, "A concise web search query."]
        ) -> str:
            if not query:
                return "Please provide a web search query."

            config = self._ensure_web_search_config()
            results = await _maybe_await(
                retrieval.search_web(
                    request=self.owui_request,
                    engine=config.WEB_SEARCH_ENGINE,
                    query=query,
                    user=self.user,
                )
            )
            return json.dumps(
                [
                    {
                        "title": result.title,
                        "url": result.link,
                        "snippet": result.snippet,
                    }
                    for result in results
                ]
            )

        await self.emit_event_safe("FinRAG is building a retrieval plan...")

        try:
            planner_output = await user_proxy.a_initiate_chat(
                recipient=planner,
                max_turns=1,
                message=f"User question: {latest_content}",
            )
            plan_content = planner_output.chat_history[-1]["content"]
            plan = json.loads(plan_content)
            steps = plan.get("steps", [])
        except Exception as exc:
            logging.warning(f"Unable to create structured FinRAG plan: {exc}")
            steps = [
                "Retrieve FinRAG project context relevant to the user question.",
                "Search filing knowledge if the question asks about a financial filing.",
            ]

        if not steps:
            steps = ["Retrieve FinRAG project context relevant to the user question."]

        if _should_use_project_context_only(latest_content):
            steps = ["Retrieve FinRAG project context relevant to the user question."]

        evidence = []
        for step in steps[:max_plan_steps]:
            await self.emit_event_safe(f"FinRAG retrieval step: {step}")
            output = await user_proxy.a_initiate_chat(
                recipient=researcher,
                max_turns=6,
                message=(
                    f"Instruction: {step}\n"
                    f"Original user question: {latest_content}\n"
                    "Return a grounded evidence summary for this step."
                ),
            )

            assistant_replies = []
            tool_outputs = []
            for chat_item in output.chat_history:
                if chat_item.get("role") == "tool":
                    tool_outputs.append(chat_item.get("content", ""))
                if chat_item.get("name") == "FinRAG_Researcher" and chat_item.get("content"):
                    assistant_replies.append(chat_item["content"])

            evidence.append(
                {
                    "step": step,
                    "answers": [] if tool_outputs else assistant_replies,
                    "tool_outputs": tool_outputs,
                }
            )

        await self.emit_event_safe("FinRAG is writing the final answer...")
        final_payload = {
            "user_question": latest_content,
            "retrieval_steps": steps[:max_plan_steps],
            "evidence": evidence,
        }
        if _asks_for_finrag_metrics(latest_content):
            final_payload["metric_guardrail"] = (
                "Use exactly these absolute metric values and deltas: "
                "nDCG@5 0.432 versus 0.371 (+0.061), "
                "MAP@5 0.391 versus 0.274 (+0.117), and "
                "MRR@5 0.640 versus 0.587 (+0.053). "
                "Do not convert +0.061, +0.117, or +0.053 into percentages."
            )

        final_output = await user_proxy.a_initiate_chat(
            recipient=report_generator,
            max_turns=1,
            message=json.dumps(final_payload, default=str),
        )
        final_content = final_output.chat_history[-1]["content"]
        return _enforce_finrag_metric_guardrail(final_content, latest_content)
