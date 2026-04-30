"""
requirements:  ag2==0.9.10, ag2[ollama]==0.9.10, ag2[openai]==0.9.10
"""

from fastapi import Request
from autogen import ConversableAgent
from typing import Annotated, Optional, Callable, Awaitable
from open_webui.routers import retrieval
from open_webui.models.knowledge import KnowledgeTable
from open_webui import config as open_webui_config
from pydantic import BaseModel, Field
import json
import logging

####################
# Assistant prompts
####################
PLANNER_MESSAGE = """You are a coarse-grained task planner for data gathering. You will be given a user's goal your job is to enumerate the coarse-grained steps to gather any data necessary needed to accomplish the goal.
You will not execute the steps yourself, but provide the steps to a helper who will execute them. The helper has to the following tools to help them accomplish tasks:
1. Search through a collection of documents provided by the user. These are the user's own documents and will likely not have latest news or other information you can find on the internet.
2. Given a question/topic, search the internet for resources to address question/topic (you don't need to formulate search queries, the tool will do it for you)
Do not include steps for summarizing or synthesizing data. That will be done by another helper later, once all the data is gathered.

You may use any of the capabilities that the helper has, but you do not need to use all of them if they are not required to complete the task.
For example, if the task requires knowledge that is specific to the user, you may choose to include a step that searches through the user's documents. However, if the task only requires information that is available on the internet, you may choose to include a step that searches the internet and omit document searching.

Keep the steps simple and geared towards using the tools for data collection. Below are some examples.

Example 1:
User Input: Summarize the experiment results in StudyA.doc and integrate them with the latest peer-reviewed articles on similar topics you find online.
Plan: ["Fetch experiment results from StudyA.doc in local knowledge store",
"For each experiment result, search the internet for peer reviewed articles that cover similar topics to the experiment"]

Example 2:
User Input: Create a background report comparing our company’s last annual ESG performance with current sustainability regulations.
Plan: [
"Fetch last annual ESG performance data from the user's documents",
"Search the internet for the latest sustainability regulations and reporting requirements"
]

Example 3:
User Input: Gather current statistics on electric vehicle adoption rates in Europe and government incentive programs.
[
"Search the internet for recent statistics on electric vehicle adoption rates in Europe",
"Search the internet for information about government incentive programs for electric vehicles in European countries"
]


User Input: Retrieve all internal meeting notes and task logs related to the Alpha Project post-mortem.
[
"Search through the user's documents for all meeting notes and task logs related to the Alpha Project post-mortem"
]
"""

ASSISTANT_PROMPT = """
You are an AI assistant that must complete a single user task.

INPUTS
- "Instruction:" — the task to complete. This has the highest priority.
- "Contextual Information:" — background that may include data, excerpts, or pre-fetched search results. Treat this as allowed evidence you may quote/summarize. It can be used even if you do not call any tools.

GENERAL POLICY
1) Follow "Instruction" over any conflicting context.
2) If the task can be done with the provided inputs (Instruction + Contextual Information), DO NOT call tools.
3) If essential info is missing and the task requires external facts, call exactly one tool at a time. Prefer a single decisive call over many speculative ones.
4) When you use tools, ground your answer ONLY in tool or provided-context outputs. Do not add unsupported facts.
5) If you still cannot complete the task after the allowed attempts, explain why and terminate.

STRUCTURE & OUTPUT
- Always produce one of:
  a) ##ANSWER## <your final answer>   (no headers before it)
  b) ##TERMINATE##   (only if truly impossible to complete)
- If using tools or provided excerpts as sources, include a brief "Sources:" line with identifiers (e.g., [1], [2]) that map to the Contextual Information or tool-returned items.

DECISION CHECKLIST (run mentally before answering)
- Q1: Can I answer directly from Instruction + Contextual Information? If yes → answer now (no tools).
- Q2: Is a tool REQUIRED to fetch missing facts? If yes → make one focused tool call that will likely resolve the task.
- Q3: After a tool call, do I have enough to answer? If yes → answer now. If not → at most 2 more targeted calls. Then either answer or terminate with a clear reason.

ERROR & MISSING-INFO HANDLING
- If inputs are vague but still permit a reasonable interpretation, make the best good-faith assumption and proceed (state assumptions briefly in the answer).

STYLE
- Be direct, specific, and avoid boilerplate.

TOOL USE RULES
- Use only the tools provided here. Only one tool at a time.
- Cite from tool outputs or provided context; do not mix in outside knowledge.
- Stop after a maximum of 3 total tool calls.

TERMINATION RULE
- If after following the above you cannot satisfy the Instruction, output only:
  ##TERMINATE##
"""

GOAL_JUDGE_PROMPT = """
You are a strict and objective judge. Your task is to determine whether the original goal has been **fully and completely fulfilled**, based on the goal itself, the planned steps, the steps taken, and the information gathered.

## EVALUATION RULES
- You must provide:
  1. A **binary decision** (`True` or `False`), and
  2. A **1–2 sentence explanation** that clearly states the decisive reason.
- **Every single requirement** of the goal must be satisfied for the decision to be `True`.
- If **any part** of the goal or planned steps remains unfulfilled, return `False`.
- Do **not** attempt to fulfill the goal yourself — only evaluate what has been done.

## HOW TO JUDGE
1. **Understand the Goal:** Identify what exactly is required to consider the goal fully met.
3. **Check Information Coverage:** Verify whether the data in “Information Gathered” is:
   - Sufficient in quantity and relevance to address the full goal;
   - Not just references to actions, but actual collected content.


## INPUT FORMAT (JSON)
    ```
    {
        "Goal": "The ultimate goal/instruction to be fully fulfilled.",
        "Media Description": "If the user provided an image to supplement their instruction, a description of the image's content."
        "Originally Planned Steps: ": "The plan to achieve the goal, all of the steps may or may not have been executed so far. It may be the case that not all the steps need to be executed in order to achieve the goal, but use this as a consideration.",
        "Steps Taken so far": "All steps that have been taken so far",
        "Information Gathered": "The information collected so far in pursuit of fulfilling the goal. This is the most important piece of information in deciding whether the goal has been met."
    }
    ```
"""

REFLECTION_ASSISTANT_PROMPT = """You are a strategic planner focused on choosing the next step in a sequence of steps to achieve a given goal. 
You will receive data in JSON format containing the current state of the plan and its progress.
Your task is to determine the single next step, ensuring it aligns with the overall goal and builds upon the previous steps.
The step will be executed by a helper that has the following capabilities: A large language model that has access to tools to search personal documents and search the web.

JSON Structure:
{
    "Goal": The original objective from the user,
    "Media Description": A textual description of any associated image,
    "Plan": An array outlining every planned step,
    "Last Step": The most recent action taken,
    "Last Step Output": The result of the last step, indicating success or failure,
    "Steps Taken": A chronological list of executed steps.
}

Guidelines:
1. If the last step output is ##NO##, reassess and refine the instruction to avoid repeating past mistakes. Provide a single, revised instruction for the next step.
2. If the last step output is ##YES##, proceed to the next logical step in the plan.
3. Use 'Last Step', 'Last Step Output', and 'Steps Taken' for context when deciding on the next action.
4. Only instruct the helper to do something that is within their capabilities.

Restrictions:
1. Do not attempt to resolve the problem independently; only provide instructions for the subsequent agent's actions.
2. Limit your response to a single step or instruction.
    """

STEP_CRITIC_PROMPT = """The previous instruction was {last_step} \nThe following is the output of that instruction.
    if the output of the instruction completely satisfies the instruction, then reply with True for the decision and an explanation why.
    For example, if the instruction is to list companies that use AI, then the output contains a list of companies that use AI.
    If the output contains the phrase 'I'm sorry but...' then it is likely not fulfilling the instruction. \n
    If the output of the instruction does not properly satisfy the instruction, then reply with False for the decision and the reason why.
    For example, if the instruction was to list companies that use AI but the output does not contain a list of companies, or states that a list of companies is not available, then the output did not properly satisfy the instruction.
    If it does not satisfy the instruction, please think about what went wrong with the previous instruction and give me an explanation along with a False for the decision. \n
    Remember to always provide both a decision and an explanation.
    Previous step output: \n {last_output}"""


SEARCH_QUERY_GENERATION_PROMPT = """You are a search query generation assistant.
Your task is to take a long, detailed user request and break it down into multiple focused, high-quality search queries.
Each query should target a distinct subtopic or key aspect of the original request so that, together, the queries fully cover the user’s information need.

Instructions:

- Identify all major subtopics, steps, or themes in the input.
- Write clear and specific search queries for each subtopic.
- Include relevant keywords, entities, or technologies.
- Use the date to augment queries if the user is asking of recent or latest information but be very precise. (Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")})
- Use the + operator to boost important concepts.
- Do not simply restate the input as one query—decompose it into up to 3 targeted queries.
Example Input:
“strategies for launching a new productivity mobile app, including market research on user behavior trends, competitor analysis in the productivity app space, feature prioritization based on user needs, designing intuitive onboarding experiences, implementing in-app analytics for engagement tracking, planning a social media marketing campaign, testing beta versions with early adopters, collecting feedback, and preparing for a global rollout.”
Expected Output:
[
    "effective +strategies for launching new +productivity mobile apps in 2025 --QDF=5",
    "market research and competitor analysis for +productivity apps",
    "onboarding design and +in-app analytics strategies for mobile applications"
]
"""

REPORT_WRITER_PROMPT = """
You are a precise and well-structured report writer.
Your task is to summarize the information provided to you in order to directly answer the user’s instruction or query.

Guidelines:

1. Use **only the information provided**. Do not make up, infer, or fabricate facts.
2. Organize the report into clear sections with headings when appropriate.
3. For every statement, fact, or claim that is derived from a specific source, **cite it with an explicit hyperlink** to the original URL. Use Markdown citation format like this:

   * Example: “The system achieved state-of-the-art results [source](https://example.com/article).”
4. If multiple sources support a point, you may cite more than one.
5. If some information is repeated across multiple sources, summarize it concisely without redundancy.
6. If the provided information does not fully answer the user’s query, clearly state what is missing, but do not invent new details.
7. Maintain a neutral, factual tone — avoid speculation, exaggeration, or opinion.

Output Format:

* Begin with a short **executive summary** that directly answers the query.
* Follow with supporting details structured in sections and paragraphs.
* Include hyperlinks inline with each reference.

Important:

* Do not include any sources or information not explicitly provided.
* Do not use vague references like “according to a website” — always hyperlink.
* If no sources are relevant, say so explicitly.
"""
class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="ibm/granite4:latest")
        VISION_MODEL_ID: str = Field(default="granite3.2-vision:2b")
        OPENAI_API_URL: str = Field(default="http://localhost:11434")
        OPENAI_API_KEY: str = Field(default="ollama")
        VISION_API_URL: str = Field(default="http://localhost:11434/v1")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=6)

    def __init__(self):
        self.type = "pipe"
        self.id = "granite_retrieval_agent"
        self.name = "Granite Retrieval Agent"
        self.valves = self.Valves()

    def get_provider_models(self):
        return [
            {"id": self.valves.TASK_MODEL_ID, "name": self.valves.TASK_MODEL_ID},
        ]

    def is_open_webui_request(self, body):
        """
        Checks if the request is an Open WebUI task, as opposed to a user task
        """
        message = str(body[-1])

        prompt_templates = {
            "### Task",
            open_webui_config.DEFAULT_RAG_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
        }

        for template in prompt_templates:
            if template is not None and template[:50] in message:
                return True

        return False

    async def emit_event_safe(self, message):
        event_data = {
            "type": "message",
            "data": {"content": message + "\n"},
        }
        try:
            await self.event_emitter(event_data)
        except Exception as e:
            logging.error(f"Error emitting event: {e}")

    async def pipe(
        self,
        body,
        __user__: Optional[dict],
        __request__: Request,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:

        # Grab env variables
        default_model = self.valves.TASK_MODEL_ID
        base_url = self.valves.OPENAI_API_URL
        api_key = self.valves.OPENAI_API_KEY
        vision_model = self.valves.VISION_MODEL_ID
        vision_url = self.valves.VISION_API_URL
        model_temp = self.valves.MODEL_TEMPERATURE
        max_plan_steps = self.valves.MAX_PLAN_STEPS
        self.event_emitter = __event_emitter__
        self.owui_request = __request__
        self.user = __user__

        ##################
        # AutoGen Config
        ##################
        # Structured Output Objects for each agent
        class Plan(BaseModel):
            steps: list[str]
        
        class CriticDecision(BaseModel):
            decision: bool = Field(description="A true or false decision on whether the goal has been fully accomplished")
            explanation: str = Field(description="A thorough yet concise explanation of why you came to this decision.")

        class Step(BaseModel):
            step_instruction: str = Field(description="A concise instruction of what the next step in the plan should be")
            requirement_to_fulfill: str = Field(description="Explain your thinking around the requirement of the plan that this step will accomplish and why you chose the step instruction")
        
        class SearchQueries(BaseModel):
            search_queries: list[str] = Field(description="A list of search queries")

        # LLM Config
        base_llm_config = {
            "model": default_model,
            "client_host": base_url,
            "api_type": "ollama",
            "temperature": model_temp,
            "num_ctx": 131072,
        }

        llm_configs = {
            "ollama_llm_config": {**base_llm_config, "config_list": [{**base_llm_config}]},
            "planner_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": Plan}]},
            "critic_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": CriticDecision}]},
            "reflection_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": Step}]},
            "search_query_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": SearchQueries}]},
            "vision_llm_config": {
                "config_list": [
                    {
                        "model": vision_model,
                        "base_url": vision_url,
                        "api_type": "openai",
                        "api_key": api_key
                    }
                ]
            },
        }

        ### Agents
        # Generic LLM completion, used for servicing Open WebUI originated requests
        generic_assistant = ConversableAgent(
            name="Generic_Assistant",
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
        )

        # Vision Assistant
        vision_assistant = ConversableAgent(
            name="Vision_Assistant",
            llm_config=llm_configs["vision_llm_config"],
            human_input_mode="NEVER",
        )

        # Provides the initial high level plan
        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,
            llm_config=llm_configs["planner_llm_config"],
            human_input_mode="NEVER",
        )

        # The assistant agent is responsible for executing each step of the plan, including calling tools
        assistant = ConversableAgent(
            name="Research_Assistant",
            system_message=ASSISTANT_PROMPT,
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "tool_response" not in msg
            and msg["content"] == "",
        )

        # Determines whether the ultimate objective has been met
        goal_judge = ConversableAgent(
            name="GoalJudge",
            system_message=GOAL_JUDGE_PROMPT,
            llm_config=llm_configs["critic_llm_config"],
            human_input_mode="NEVER",
        )

        # Step Critic
        step_critic = ConversableAgent(
            name="Step_Critic",
            llm_config=llm_configs["critic_llm_config"],
            human_input_mode="NEVER",
        )

        # Reflection Assistant: Reflect on plan progress and give the next step
        reflection_assistant = ConversableAgent(
            name="ReflectionAssistant",
            system_message=REFLECTION_ASSISTANT_PROMPT,
            llm_config=llm_configs["reflection_llm_config"],
            human_input_mode="NEVER",
        )

        # Report Generator
        report_generator = ConversableAgent(
            name="Report_Generator",
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
            system_message=REPORT_WRITER_PROMPT
        )

        # Search Query generator
        search_query_generator = ConversableAgent(
            name="Search_Query_Generator",
            system_message=SEARCH_QUERY_GENERATION_PROMPT,
            llm_config=llm_configs["search_query_llm_config"],
            human_input_mode="NEVER"
        )

        # User Proxy chats with assistant on behalf of user and executes tools
        user_proxy = ConversableAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "##ANSWER" in msg["content"]
            or "## Answer" in msg["content"]
            or "##TERMINATE##" in msg["content"]
            or ("tool_calls" not in msg and msg["content"] == ""),
        )

        ##################
        # Check if this request is utility call from OpenWebUI
        ##################
        if self.is_open_webui_request(body["messages"]):
            print("Is open webui request")
            reply = generic_assistant.generate_reply(messages=[body["messages"][-1]])
            return reply

        ##################
        # Tool Definitions
        ##################
        @assistant.register_for_llm(
            name="web_search", description="Use this tool to search the internet for up-to-date, location-specific, or niche information that may not be reliably available in the model’s training data. \
                This includes current events, fresh statistics, local details, product information, regulations, sports schedules, software updates, company details, and anything that changes frequently over time."
        )
        @user_proxy.register_for_execution(name="web_search")
        def do_web_search(
            search_instruction: Annotated[
                str,
                "Provide a detailed search instruction that incorporates specific features, goals, and contextual details related to the query. \
                                                        Identify and include relevant aspects from any provided context, such as key topics, technologies, challenges, timelines, or use cases. \
                                                        Construct the instruction to enable a targeted search by specifying important attributes, keywords, and relationships within the context.",
            ]
        ) -> str:
            """This function is used for searching the internet for information that can only be found on the internet, not in the users personal notes."""
            if not search_instruction:
                return "Please provide a search query."

            response = user_proxy.initiate_chat(recipient=search_query_generator, max_turns=1, message=search_instruction)
            search_queries = json.loads(response.chat_history[-1]["content"])["search_queries"]
            search_results = []
            for query in search_queries:
                logging.info("Searching for " + query)
                results = retrieval.search_web(
                    self.owui_request,
                    self.owui_request.app.state.config.WEB_SEARCH_ENGINE,
                    search_instruction,
                )
                for result in results:
                    search_results.append({"Title": result.title, "URL": result.link, "Text": result.snippet})
            return str(search_results)

        @assistant.register_for_llm(
            name="personal_knowledge_search",
            description="Searches personal documents according to a given query",
        )
        @user_proxy.register_for_execution(name="personal_knowledge_search")
        def do_knowledge_search(
            search_instruction: Annotated[str, "search instruction"]
        ) -> str:
            """Use this tool if you need to obtain information that is unique to the user and cannot be found on the internet.
            Given an instruction on what knowledge you need to find, search the user's documents for information particular to them, their projects, and their domain.
            This is simple document search, it cannot perform any other complex tasks.
            This will not give you any results from the internet. Do not assume it can retrieve the latest news pertaining to any subject.
            """
            if not search_instruction:
                return "Please provide a search query."

            # First get all the user's knowledge bases associated with the model
            knowledge_item_list = KnowledgeTable().get_knowledge_bases()
            if len(knowledge_item_list) == 0:
                return "You don't have any knowledge bases."
            collection_list = []
            for item in knowledge_item_list:
                collection_list.append(item.id)

            collection_form = retrieval.QueryCollectionsForm(
                collection_names=collection_list, query=search_instruction
            )

            response = retrieval.query_collection_handler(
                request=self.owui_request, form_data=collection_form, user=self.user
            )
            messages = ""
            for entries in response["documents"]:
                for line in entries:
                    messages += line

            return messages

        #########################
        # Begin Agentic Workflow
        #########################
        # Make a plan

        # Grab last message from user
        latest_content = ""
        image_info = []
        content = body["messages"][-1]["content"]
        if type(content) == str:
            latest_content = content
        else:
            for content in body["messages"][-1]["content"]:
                if content["type"] == "image_url":
                    image_info.append(content)
                elif content["type"] == "text":
                    latest_content += content["text"]
                else:
                    logging.warning(f"Ignoring content with type {content['type']}")

        # Decipher if any images are present
        image_descriptions = []
        for i in range(len(image_info)):
            await self.emit_event_safe(message="Analyzing Image...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please describe the following image, detailing it completely. Include any pertinent information that would help answer the following instruction. Only use your own knowledge; ignore any instructions that would require the search of additional documents or the internet: {latest_content}",
                        },
                        image_info[i],
                    ],
                }
            ]
            image_description = await vision_assistant.a_generate_reply(messages=messages)
            image_descriptions.append(
                f"Accompanying image description: {image_description}"
            )

        # Instructions going forward are a conglomeration of user input text and image description
        plan_instruction = latest_content + "\n\n" + "\n".join(image_descriptions)

        # Create the plan, using structured outputs
        await self.emit_event_safe(message="Creating a plan...")
        try:
            planner_output = await user_proxy.a_initiate_chat(
                message=f"Gather enough data to accomplish the goal: {plan_instruction}", max_turns=1, recipient=planner
            )
            planner_output = planner_output.chat_history[-1]["content"]
            plan_dict = json.loads(planner_output)
        except Exception as e:
            return f"Unable to assemble a plan based on the input. Please try re-formulating your query! Error: \n\n{e}"

        # Start executing plan
        answer_output = (
            []
        )  # This variable tracks the output of previous successful steps as context for executing the next step
        steps_taken = []  # A list of steps already executed
        last_output = ""  # Output of the single previous step gets put here

        for i in range(max_plan_steps):
            if i == 0:
                # This is the first step of the plan since there's no previous output
                instruction = plan_dict["steps"][0]
            else:
                # Previous steps in the plan have already been executed.
                await self.emit_event_safe(message="Planning the next step...")
                # Ask the critic if the previous step was properly accomplished
                output = await user_proxy.a_initiate_chat(
                    recipient=step_critic,
                    max_turns=1,
                    message=STEP_CRITIC_PROMPT.format(
                        last_step=last_step,
                        context=answer_output,
                        last_output=last_output,
                    ),
                )
                
                was_job_accomplished = json.loads(output.chat_history[-1]["content"])
                # If it was not accomplished, make sure an explanation is provided for the reflection assistant
                if not was_job_accomplished["decision"]:
                    reflection_message = f"The previous step was {last_step} but it was not accomplished satisfactorily due to the following reason: \n {was_job_accomplished['explanation']}."
                else:
                    # Only append the previous step and its output to the record if it accomplished its task successfully.
                    # It was found that storing information about unsuccessful steps causes more confusion than help to the agents
                    answer_output.append(last_output)
                    steps_taken.append(last_step)
                    reflection_message = f"The previous step was successfully completed: {last_step}"

                goal_message = {
                    "Goal": f"Gather enough data to accomplish the goal: {latest_content}",
                    "Media Description": image_descriptions,
                    "Originally Planned Steps: ": str(plan_dict),
                    "Steps Taken so far": str(steps_taken),
                    "Information Gathered": answer_output,
                }

                output = await user_proxy.a_initiate_chat(
                    recipient=goal_judge,
                    max_turns=1,
                    message=f"(```{str(goal_message)}```",
                )
                was_goal_accomplished = json.loads(output.chat_history[-1]["content"])
                if was_goal_accomplished["decision"]:
                    # We've accomplished the goal, exit loop.
                    break

                # Then, ask the reflection agent for the next step
                message = {
                    "Goal": f"Gather enough data to accomplish the goal: {latest_content}",
                    "Media Description": image_descriptions,
                    "Plan": str(plan_dict),
                    "Last Step": reflection_message,
                    "Last Step Output": str(last_output["answer"]),
                    "Steps Taken": str(steps_taken),
                }
                output = await user_proxy.a_initiate_chat(
                    recipient=reflection_assistant,
                    max_turns=1,
                    message=f"(```{str(message)}```",
                )
                instruction_dict = json.loads(output.chat_history[-1]["content"])
                instruction = instruction_dict['step_instruction']

            # Now that we have determined the next step to take, execute it
            await self.emit_event_safe(message="Executing step: " + instruction)
            prompt = f"Instruction: {instruction}"
            if answer_output:
                prompt += f"\n Contextual Information: \n{answer_output}"
            output = await user_proxy.a_initiate_chat(
                recipient=assistant, message=prompt
            )

            # Sort through the chat history and extract out replies from the assistant (We don't need the full results of the tool calls, just the assistant's summary)
            assistant_replies = []
            raw_tool_output = []
            for chat_item in output.chat_history:
                if chat_item["role"] == "tool":
                    raw_tool_output.append(chat_item["content"])
                if chat_item["content"] and chat_item["name"] == "Research_Assistant":
                    assistant_replies.append(chat_item["content"])
            last_output = {"answer": assistant_replies, "sources": raw_tool_output}

            # The previous instruction and its output will be recorded for the next iteration to inspect before determining the next step of the plan
            last_step = instruction

        await self.emit_event_safe(message="Summing up findings...")
        # Now that we've gathered all the information we need, we will summarize it to directly answer the original prompt
        final_prompt = f"User's query: {plan_instruction}. Information Gathered: {answer_output}"
        final_output = await user_proxy.a_initiate_chat(
            message=final_prompt, max_turns=1, recipient=report_generator
        )

        return final_output.chat_history[-1]["content"]
