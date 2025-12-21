import os
import torch
from typing import List, Dict, Optional
from pydantic import ValidationError

from unsloth import FastLanguageModel  # ✅ 新增：Unsloth

from smolagents.models import MessageRole, get_clean_message_list

# 你上面已有
openai_role_conversions = {MessageRole.TOOL_RESPONSE: MessageRole.USER}


def apply_stop_sequences(generated_text: str, stop_sequences: List[str]) -> str:
    """
    Cut off the model output at the first occurrence of any stop sequence.
    If none appear, return full text.
    """
    if not stop_sequences:
        return generated_text

    cut_positions = []
    for stop in stop_sequences:
        idx = generated_text.find(stop)
        if idx != -1:
            cut_positions.append(idx)

    if len(cut_positions) == 0:
        return generated_text

    first_cut = min(cut_positions)
    return generated_text[:first_cut].rstrip()


def llama_format_messages_for_tokenizer(messages: List[Dict], tokenizer) -> str:
    """
    Turn a list of {role, content} into a chat-style prompt that Llama expects.

    We'll try tokenizer.apply_chat_template if it exists (Llama-3-style).
    Fallback: simple manual concat.

    Expected roles in `messages`: "system", "user", "assistant".
    We assume you've already run get_clean_message_list() so roles are normalized.
    """
    # Newer Llama models on HF (like Llama 3 Instruct) ship a chat_template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # HF expects a list of dicts like [{"role":"user","content":"hi"}, ...]
        # We just feed it directly.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # means "assistant:" at the end
        )

    # Fallback template if model/tokenizer doesn't have chat_template
    # Very plain: [SYSTEM] ... \n[USER] ... \n[ASSISTANT]
    system_prefix = ""
    user_assistant_turns = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_prefix += f"<<SYS>>\n{content}\n<</SYS>>\n"
        elif role == "user":
            user_assistant_turns.append(f"[USER]: {content}")
        elif role == "assistant":
            user_assistant_turns.append(f"[ASSISTANT]: {content}")
        else:
            # default to user if it's some tool_response remapped
            user_assistant_turns.append(f"[USER]: {content}")

    # Add final assistant tag as generation cue
    prompt = system_prefix + "\n".join(user_assistant_turns) + "\n[ASSISTANT]: "
    return prompt


class LlamaEngine:
    """
    Local Llama inference engine using Unsloth (FastLanguageModel).

    Goal:
    - Same high-level API as your OpenAIEngine/AnthropicEngine:
        __call__(messages, stop_sequences=[])
    - messages: list[{"role": "...", "content": "..."}]
    - return: str (assistant's reply)
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16,
        max_new_tokens: int = 8192,
    ):
        """
        model_name: HF model id or local path, e.g. "meta-llama/Llama-3-8B-Instruct"
        device: "cuda", "cuda:0", "cpu", etc. If None, picks cuda if available.
        dtype: torch.float16 / bfloat16 / float32 etc.
        max_new_tokens: default generation length
        """

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # device 只用来放 inputs，模型由 Unsloth 自己管理
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # -------- Unsloth: from_pretrained --------
        # 将 torch.dtype 转成 Unsloth 习惯的字符串（也可以直接用 None，让 Unsloth 自适应）
        if isinstance(dtype, torch.dtype):
            if dtype == torch.float16:
                unsloth_dtype = "float16"
            elif dtype == torch.bfloat16:
                unsloth_dtype = "bfloat16"
            elif dtype == torch.float32:
                unsloth_dtype = "float32"
            else:
                unsloth_dtype = None
        else:
        # 如果用户直接传字符串，原样给 Unsloth
            unsloth_dtype = dtype

        # 如果是 GPU，默认用 4bit；CPU 上就不开 4bit
        load_in_4bit = "cuda" in self.device

        # 这里给个常用 max_seq_length；你需要的话可以改成参数
        max_seq_length = 4096

        # ✅ 用 Unsloth 加载模型 & tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=unsloth_dtype,
            load_in_4bit=load_in_4bit,
        )

        # 设定为 inference 模式（会关掉 dropout 等）
        FastLanguageModel.for_inference(self.model)

        # optional metrics collection similar to AzureOpenAIEngine
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def __call__(
        self,
        messages: List[Dict],
        stop_sequences: List[str] = [],
        temperature: float = 0.5,
        max_new_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a reply from the local Llama model (via Unsloth).

        messages: [{"role":"system"/"user"/"assistant", "content": "..."}]
        stop_sequences: list of strings -> we'll manually truncate after generation
        temperature, top_p: sampling params
        max_new_tokens: override default length
        """
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        prompt_text = llama_format_messages_for_tokenizer(messages, self.tokenizer)

        # 1) determine model max context
        max_prompt_tokens = 4000

        # 4) tokenize with truncation
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_token_count = inputs["input_ids"].shape[-1]
        
        # 4) generate （Unsloth 的模型接口跟 HF 一样）
        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
        
        # 5) decode & apply stop sequences
        # ✅ 用 token 级别来切分：先取出生成的所有 token
        generated_ids = gen_out[0]
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[input_len:]
        # 再 decode 这部分
        completion_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_text = apply_stop_sequences(completion_text, stop_sequences)

        # Update metrics like AzureOpenAIEngine
        completion_token_count = self.tokenizer(
            completion_text,
            return_tensors="pt"
        )["input_ids"].shape[-1]

        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += prompt_token_count
        self.metrics["completion_tokens"] += completion_token_count

        return completion_text.strip()

    def reset(self):
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }




if __name__ == "__main__":
    print("=== Testing LlamaEngine with Unsloth + Llama 3.1 8B Instruct ===")

    # 你指定的模型
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2-7B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    engine = LlamaEngine(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 测试消息格式
    messages = [
        {"role": "system", "content": "You are a concise and helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    print(f"\n--- Sending message to {model_name} ---")
    output = engine(
        messages,
        temperature=0.1,
    )

    print("\n=== Model Output ===")
    print(output)

    print("\n=== Metrics After First Call ===")
    print(engine.metrics)