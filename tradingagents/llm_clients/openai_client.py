import os
from typing import Any, Mapping, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .codex_oauth import (
    CODEX_RESPONSES_BASE_URL,
    codex_oauth_default_headers,
    codex_oauth_requested,
    load_codex_oauth_credentials,
)
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling. ``with_structured_output`` defaults
    to function-calling so the Responses-API parse path is avoided
    (langchain-openai's parse path emits noisy
    PydanticSerializationUnexpectedValue warnings per call without
    affecting correctness).

    Provider-specific quirks (e.g. DeepSeek's thinking mode) live in
    purpose-built subclasses below so this base class stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


_CODEX_UNSUPPORTED_PARAMS = (
    "ls_structured_output_format",
    "max_output_tokens",
    "metadata",
    "prompt_cache_retention",
    "structured_output_format",
    "temperature",
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _normalize_codex_content(content: Any, *, assistant: bool = False) -> Any:
    text_type = "output_text" if assistant else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        text = str(content) if content is not None else ""
        return [{"type": text_type, "text": text}]

    normalized = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": text_type, "text": block})
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            normalized.append({"type": text_type, "text": block.get("text", "")})
        elif block_type == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            entry = {"type": "input_image", "image_url": str(url or "")}
            detail = image_url.get("detail") if isinstance(image_url, Mapping) else None
            if detail:
                entry["detail"] = detail
            normalized.append(entry)
        elif block_type in {
            "input_text",
            "input_image",
            "input_file",
            "output_text",
            "refusal",
        }:
            normalized.append(dict(block))
    return normalized


def _normalize_codex_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    instructions = []
    normalized_input = []

    for item in payload.get("input") or []:
        if not isinstance(item, Mapping):
            normalized_input.append(item)
            continue
        item_dict = dict(item)
        role = item_dict.get("role")
        if role in {"system", "developer"}:
            text = _content_to_text(item_dict.get("content"))
            if text:
                instructions.append(text)
            continue
        if item_dict.get("type") == "message" or role in {"user", "assistant"}:
            item_dict["type"] = "message"
            item_dict["content"] = _normalize_codex_content(
                item_dict.get("content"),
                assistant=(role == "assistant"),
            )
        normalized_input.append(item_dict)

    existing_instructions = _content_to_text(payload.get("instructions"))
    if existing_instructions or instructions:
        payload["instructions"] = "\n\n".join(
            part for part in [existing_instructions, *instructions] if part
        )
    elif not existing_instructions:
        payload["instructions"] = "You are a helpful assistant."

    payload["input"] = normalized_input or [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": " "}],
        }
    ]
    payload["stream"] = True
    payload["store"] = False

    for key in _CODEX_UNSUPPORTED_PARAMS:
        payload.pop(key, None)

    if "tool_choice" not in payload and payload.get("tools"):
        payload["tool_choice"] = "auto"

    return payload


class CodexOAuthChatOpenAI(NormalizedChatOpenAI):
    """ChatOpenAI variant that targets the ChatGPT Codex Responses backend."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _normalize_codex_responses_payload(payload)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Two quirks that don't apply to other OpenAI-compatible providers:

    1. **Thinking-mode round-trip.** When DeepSeek's thinking models return
       a response with ``reasoning_content``, that field must be echoed
       back as part of the assistant message on the next turn or the API
       fails with HTTP 400. ``_create_chat_result`` captures the field on
       receive and ``_get_request_payload`` re-attaches it on send.

    2. **deepseek-reasoner has no tool_choice.** Structured output via
       function-calling is unavailable, so we raise NotImplementedError
       and let the agent factories fall back to free-text generation
       (see ``tradingagents/agents/utils/structured.py``).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError(
                "deepseek-reasoner does not support tool_choice; structured "
                "output is unavailable. Agent factories fall back to "
                "free-text generation automatically."
            )
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "service_tier",
    "api_key", "callbacks", "http_client", "http_async_client",
    "default_headers",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        use_codex_oauth = self.provider == "openai" and codex_oauth_requested()

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_CONFIG:
            default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = self.base_url or default_base
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if use_codex_oauth:
            credentials = load_codex_oauth_credentials()
            llm_kwargs["base_url"] = CODEX_RESPONSES_BASE_URL
            llm_kwargs["api_key"] = credentials.access_token
            llm_kwargs["default_headers"] = codex_oauth_default_headers(credentials)
            llm_kwargs["use_responses_api"] = True
            llm_kwargs["streaming"] = True
        elif self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        # DeepSeek's thinking-mode quirks live in their own subclass so the
        # base NormalizedChatOpenAI stays free of provider-specific branches.
        if use_codex_oauth:
            chat_cls = CodexOAuthChatOpenAI
        elif self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
