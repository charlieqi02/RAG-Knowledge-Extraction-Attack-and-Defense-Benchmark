import os
from time import sleep

import openai
# from google import genai
from anthropic import Anthropic, AnthropicVertex
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from smolagents.models import MessageRole, get_clean_message_list


openai_role_conversions = {MessageRole.TOOL_RESPONSE: MessageRole.USER}


class OpenAIEngine:
    def __init__(self, model_name="gpt-4o", api_key=None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        

    def __call__(self, messages, stop_sequences=[], temperature=0.5, *args, **kwargs):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=temperature,
            # store=True,
            # metadata =  {
            #     "user": "ZhishengQi",
            #     "project": "Knowledge-Extraction-Attacks-and-Defenses-on-RAG",
            # },
            *args,
            **kwargs,
        )
        
        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += response.usage.prompt_tokens
        self.metrics["completion_tokens"] += response.usage.completion_tokens
        
        return response.choices[0].message.content
    
    def reset(self):
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }



class AnthropicVertexEngine:
    def __init__(
        self, model_name: str = "claude-3-5-sonnet-v2@20241022",
        project_id: str = "", region: str = ""
    ):
        self.model_name = model_name
        self.project_id = project_id
        self.region = region

        self.client = AnthropicVertex(region=self.region, project_id=self.project_id)

    def __call__(
        self, messages, stop_sequences=None, temperature: float = 0, max_tokens: int = 4096,
        *args, **kwargs,
    ):
        if stop_sequences is None:
            stop_sequences = []

        # Normalize messages (same as your other engines)
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        # Extract system prompt (keep your current logic)
        index_system_message, system_prompt = "", ""
        for index, message in enumerate(messages):
            if message["role"] == MessageRole.SYSTEM:
                index_system_message = index
                system_prompt = message["content"]
                break

        filtered_messages = [m for i, m in enumerate(messages) if i != index_system_message]
        if len(filtered_messages) == 0:
            raise Exception(f"Error: no non-system message found! messages={messages}")

        # Anthropic expects roles "user"/"assistant" (and possibly tool roles depending on SDK).
        # We'll pass through user/assistant, and coerce anything else to "user" to be safe.
        vertex_messages = []
        for m in filtered_messages:
            role = m.get("role")
            if role not in (MessageRole.USER, MessageRole.ASSISTANT):
                role = MessageRole.USER
            vertex_messages.append({"role": role, "content": m["content"]})

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=vertex_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            *args,
            **kwargs,
        )

        # In Anthropic SDK, response.content is a list of blocks; text blocks have .text
        full_response_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                full_response_text += block.text

        return full_response_text


class AzureOpenAIEngine:
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = None,
    ):
        self.model_name = model_name or os.getenv("AZURE_MODEL_NAME")
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_API_KEY"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_ENDPOINT"),
            api_version=api_version or os.getenv("AZURE_API_VERSION"),
        )
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def __call__(self, messages, stop_sequences=[], temperature=0.5, *args, **kwargs):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        success = False
        wait_time = 1
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # stop=stop_sequences,
                    temperature=temperature,
                    store=True,
                    metadata =  {
                        "user": "ZhishengQi",
                        "project": "Knowledge-Extraction-Attacks-and-Defenses-on-RAG",
                    },
                    *args,
                    **kwargs
                )
                success = True
            except openai.InternalServerError:
                sleep(wait_time)
                wait_time += 1

        # Update metrics
        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += response.usage.prompt_tokens
        self.metrics["completion_tokens"] += response.usage.completion_tokens

        return response.choices[0].message.content

    def reset(self):
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


class ThoughtCodeFormat(BaseModel):
    thought: str
    code: str


class ThoughtActionFormat(BaseModel):
    thought: str
    action: str


class StructuredOutputAzureOpenAIEngine(AzureOpenAIEngine):
    def __init__(self, response_format: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_format_str = response_format
        if response_format == "thought_code":
            self.response_format = ThoughtCodeFormat
        elif response_format == "thought_action":
            self.response_format = ThoughtActionFormat

    def __call__(self, messages, temperature=0.5, stop_sequences=None, *args, **kwargs) -> dict:
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=self.response_format,
            temperature=temperature,
            *args,
            **kwargs
        )

        # Update metrics
        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += response.usage.prompt_tokens
        self.metrics["completion_tokens"] += response.usage.completion_tokens

        return response.choices[0].message.parsed



# class GeminiEngine:
#     def __init__(
#         self, model_name: str = "gemini-3-flash", project_id="", location=""):
#         self.model_name = model_name
#         self.project_id = project_id
#         self.location = location

#         # Vertex AI client (uses Application Default Credentials)
#         self.client = genai.Client(
#             vertexai=True,
#             project=self.project_id,
#             location=self.location,
#         )

#         self.metrics = {
#             "num_calls": 0,
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#         }

#     def __call__(self, messages, stop_sequences=None, temperature=0.5, *args, **kwargs):
#         if stop_sequences is None:
#             stop_sequences = []

#         # Normalize messages (same as OpenAI / Anthropic engines)
#         messages = get_clean_message_list(messages)

#         # Gemini expects a flat list of contents (system + user merged)
#         contents = []
#         for msg in messages:
#             if msg["role"] == MessageRole.SYSTEM:
#                 contents.append(f"[SYSTEM]\n{msg['content']}")
#             else:
#                 contents.append(msg["content"])

#         response = self.client.models.generate_content(
#             model=self.model_name,
#             contents=contents,
#             generation_config={
#                 "temperature": temperature,
#                 "stop_sequences": stop_sequences,
#             },
#             *args,
#             **kwargs,
#         )

#         # Update metrics if available
#         if hasattr(response, "usage_metadata"):
#             usage = response.usage_metadata
#             self.metrics["num_calls"] += 1
#             self.metrics["prompt_tokens"] += usage.prompt_token_count or 0
#             self.metrics["completion_tokens"] += usage.candidates_token_count or 0
#         else:
#             self.metrics["num_calls"] += 1

#         return response.text

#     def reset(self):
#         self.metrics = {
#             "num_calls": 0,
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#         }