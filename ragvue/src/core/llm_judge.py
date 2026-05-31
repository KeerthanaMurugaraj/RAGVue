"""Provider-agnostic LLM judge for RAGVue metrics.

Set RAGVUE_JUDGE_PROVIDER=anthropic to use Claude models instead of OpenAI.
Default provider: openai.

Environment variables:
  RAGVUE_JUDGE_PROVIDER   openai (default) | anthropic
  OPENAI_API_KEY          Required for openai provider
  ANTHROPIC_API_KEY       Required for anthropic provider
  OPENAI_BASE_URL         Optional OpenAI base URL override
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List


# ── Env loading ──────────────────────────────────────────────────────────────

def ensure_env() -> None:
    """Load API keys from .env files (both OPENAI_API_KEY and ANTHROPIC_API_KEY)."""
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
        # parents[3] = project root from ragvue/src/core/llm_judge.py
        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
        load_dotenv(Path(__file__).resolve().parents[3] / ".env.local", override=True)
    except Exception:
        pass
    # Fallback: raw line parse
    for key_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(key_name):
            for p in [
                Path.cwd() / ".env",
                Path(__file__).resolve().parents[3] / ".env",
                Path.home() / ".env",
            ]:
                if p.exists():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if line.strip().startswith(f"{key_name}="):
                            os.environ[key_name] = line.split("=", 1)[1].strip().strip("'\"")
                            break
                if os.getenv(key_name):
                    break


# ── Provider helpers ─────────────────────────────────────────────────────────

def default_model() -> str:
    """Return the default model for the active provider."""
    provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
    return "claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini"


def call_judge(messages: List[dict], model: str | None = None, temperature: float = 0.0) -> str:
    """Call the LLM judge and return the raw text response (JSON mode).

    Routes to OpenAI or Anthropic based on RAGVUE_JUDGE_PROVIDER.
    OpenAI uses response_format=json_object; Anthropic relies on prompt instructions.
    """
    provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
    m = model or default_model()
    if provider == "anthropic":
        return _call_anthropic(messages, m, temperature)
    return _call_openai(messages, m, temperature)


def call_judge_text(messages: List[dict], model: str | None = None, temperature: float = 0.0) -> str:
    """Call the LLM judge for plain-text output (no JSON response_format).

    Used for non-JSON outputs such as aspect extraction.
    """
    provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
    m = model or default_model()
    if provider == "anthropic":
        return _call_anthropic(messages, m, temperature)
    return _call_openai_text(messages, m, temperature)


def call_judge_vision(
    messages: List[dict],
    image_b64: str,
    media_type: str,
    user_text: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Call the LLM judge with an image attachment (vision).

    Injects image + user_text as the final user message.
    Supports both OpenAI and Anthropic vision APIs.
    """
    provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
    m = model or default_model()

    # All prior messages except the final turn
    prior = [msg for msg in messages if msg["role"] != "user" or msg == messages[0]]
    # Use everything up to (not including) the image turn as context
    context_msgs = list(messages)

    if provider == "anthropic":
        return _call_anthropic_vision(context_msgs, image_b64, media_type, user_text, m, temperature)
    return _call_openai_vision(context_msgs, image_b64, media_type, user_text, m, temperature)


# ── Provider implementations ─────────────────────────────────────────────────

def _call_openai(messages: List[dict], model: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return out.choices[0].message.content or ""


def _call_openai_text(messages: List[dict], model: str, temperature: float) -> str:
    """OpenAI call without JSON response_format (for free-text outputs)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return out.choices[0].message.content or ""


def _call_anthropic(messages: List[dict], model: str, temperature: float) -> str:
    """Anthropic messages.create call. Extracts system message automatically."""
    import anthropic
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kwargs: dict = {
        "model": model,
        "max_tokens": 2048,
        "messages": user_messages,
        "temperature": temperature,
    }
    if system_msg:
        kwargs["system"] = system_msg
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def _call_openai_vision(
    messages: List[dict], image_b64: str, media_type: str, user_text: str, model: str, temperature: float
) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    msgs = [m for m in messages if m["role"] == "system"]
    for m in messages:
        if m["role"] != "system":
            msgs.append(m)
    msgs.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
        ],
    })
    out = client.chat.completions.create(model=model, messages=msgs, temperature=temperature)
    return out.choices[0].message.content or ""


def _call_anthropic_vision(
    messages: List[dict], image_b64: str, media_type: str, user_text: str, model: str, temperature: float
) -> str:
    import anthropic
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    user_messages.append({
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": user_text},
        ],
    })
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kwargs: dict = {
        "model": model,
        "max_tokens": 2048,
        "messages": user_messages,
        "temperature": temperature,
    }
    if system_msg:
        kwargs["system"] = system_msg
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


# ── Streaming variants ────────────────────────────────────────────────────────

def call_judge_text_stream(messages: List[dict], model: str | None = None, temperature: float = 0.7):
    """Streaming version of call_judge_text. Yields text chunks as a generator.

    Routes to OpenAI or Anthropic based on RAGVUE_JUDGE_PROVIDER.
    """
    provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
    m = model or default_model()
    if provider == "anthropic":
        yield from _call_anthropic_stream(messages, m, temperature)
    else:
        yield from _call_openai_text_stream(messages, m, temperature)


def _call_openai_text_stream(messages: List[dict], model: str, temperature: float):
    """OpenAI streaming call — yields text delta chunks."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _call_anthropic_stream(messages: List[dict], model: str, temperature: float):
    """Anthropic streaming call — yields text chunks via messages.stream."""
    import anthropic
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kwargs: dict = {
        "model": model,
        "max_tokens": 2048,
        "messages": user_messages,
        "temperature": temperature,
    }
    if system_msg:
        kwargs["system"] = system_msg
    with client.messages.stream(**kwargs) as stream:
        yield from stream.text_stream
