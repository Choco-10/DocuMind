import os
import torch
from dotenv import load_dotenv
from threading import Lock
from typing import Optional, Generator, List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM3-3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


LOCAL_FILES_ONLY = (
    _env_true("LOCAL_FILES_ONLY")
    or _env_true("HF_HUB_OFFLINE")
    or _env_true("TRANSFORMERS_OFFLINE")
)

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_model_lock = Lock()


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return

    with _model_lock:

        if _model is not None:
            return

        use_cuda = torch.cuda.is_available()

        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                use_fast=False,
                local_files_only=LOCAL_FILES_ONLY
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for '{MODEL_NAME}'. "
                "If running offline, first download model files with internet once, "
                "then set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. "
                f"Original error: {e}"
            ) from e

        try:
            if use_cuda:
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    quantization_config=_bnb_config(),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=LOCAL_FILES_ONLY,
                )
            else:
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=LOCAL_FILES_ONLY,
                ).to("cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{MODEL_NAME}'. "
                "If running offline, first download model files with internet once, "
                "then set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. "
                f"Original error: {e}"
            ) from e

        _model.eval()


def build_messages(question: str, context: str, history_messages: Optional[List[Dict[str, str]]] = None):

    system_prompt = (
        "/no_think\n"
        "Answer using ONLY the provided context.\n"
        "The context is organized as labeled evidence blocks. Use the most relevant blocks and ignore the rest.\n"
        "If the context does not contain the answer, respond exactly:\n"
        "I don't know."
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for msg in history_messages or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    return messages


import re

_TRANSCRIPT_MARKER_RE = re.compile(r"\b(?:user|assistant|system)\s*:", flags=re.IGNORECASE)


def sanitize_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return "I don't know."

    marker = _TRANSCRIPT_MARKER_RE.search(text)
    if marker:
        text = text[:marker.start()].strip()

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "I don't know."

    if "i don't know" in text.lower():
        return "I don't know."

    return text

def clean_output(decoded: str, prompt: str) -> str:
    text = decoded

    parts = text.split("assistant")
    if len(parts) > 1:
        text = parts[-1].strip()
    else:
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

    return sanitize_generated_text(text)


def generate_answer(
    question: str,
    context: str,
    history_messages: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 128
) -> str:

    load_model()

    messages = build_messages(question, context, history_messages)

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt"
    ).to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id
        )

    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)

    answer = clean_output(decoded, prompt)

    return answer if answer else "I don't know."


def stream_answer(
    question: str,
    context: str,
    history_messages: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 128
) -> Generator[str, None, None]:

    load_model()

    messages = build_messages(question, context, history_messages)

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt"
    ).to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.95,
        do_sample=True,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id
    )

    import threading

    thread = threading.Thread(
        target=_model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    for token in streamer:
        yield token