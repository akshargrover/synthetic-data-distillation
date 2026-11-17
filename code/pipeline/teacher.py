"""
Teacher LLM wrapper.

Two interchangeable back-ends:
  * "hf"     - a local Hugging Face instruct model (default: Qwen2.5-3B-Instruct,
               4-bit NF4 so it runs on a 6 GB laptop GPU).
  * "openai" - any OpenAI-compatible chat endpoint (Ollama, LM Studio, vLLM,
               Groq, ...). Lets the same pipeline use Llama-3 / Mistral.

Both expose  .chat(list_of_message_lists, **gen_kwargs) -> list[str]
"""
from __future__ import annotations

import time

import config as C


class HFTeacher:
    def __init__(self, model_id: str = C.TEACHER_MODEL, load_4bit: bool = C.TEACHER_LOAD_4BIT):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"           # required for batched generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = dict(device_map="cuda:0", torch_dtype=torch.float16)
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.name = model_id + (" (4-bit NF4)" if load_4bit else "")

    def chat(self, conversations, max_new_tokens=256, temperature=0.0, top_p=1.0, seed=None):
        torch = self.torch
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in conversations]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None, top_k=None,
                pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


class OpenAITeacher:
    def __init__(self, base_url: str = C.OPENAI_BASE_URL, model: str = C.OPENAI_MODEL):
        from openai import OpenAI
        import os
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("TEACHER_API_KEY", "not-needed"))
        self.model = model
        self.name = model

    def chat(self, conversations, max_new_tokens=256, temperature=0.0, top_p=1.0, seed=None):
        outs = []
        for msgs in conversations:
            for attempt in range(5):
                try:
                    r = self.client.chat.completions.create(
                        model=self.model, messages=msgs, max_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p, seed=seed)
                    outs.append(r.choices[0].message.content or "")
                    break
                except Exception:          # rate limit / transient error
                    time.sleep(2 ** attempt)
            else:
                outs.append("")
        return outs


def load_teacher():
    if C.TEACHER_BACKEND == "hf":
        return HFTeacher()
    if C.TEACHER_BACKEND == "openai":
        return OpenAITeacher()
    raise ValueError(C.TEACHER_BACKEND)
