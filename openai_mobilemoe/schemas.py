"""Request models for the OpenAI-compatible surface.

Models are deliberately permissive (``extra="allow"``): OpenAI clients send a
long tail of parameters, and rejecting unknown fields would break them for no
benefit. Fields the model cannot honour are reported in ``x_mobilemoe.warnings``
(see :func:`openai_mobilemoe.translate.collect_warnings`) instead of erroring.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    response_format: Any = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    """Legacy ``/v1/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: Any = ""
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None


def error_body(
    message: str,
    kind: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """An OpenAI-shaped error envelope."""
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}
