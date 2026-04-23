from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderContext:
    uid: str
    user_message: str
    history: list[dict[str, Any]]
    upload_name: str | None
    upload_content_type: str | None
    upload_payload: bytes | None
    max_tool_rounds: int = 8


@dataclass
class ProviderEvent:
    kind: str
    text: str | None = None
    model: str | None = None
    citations: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    detail: str | None = None


ToolDispatch = Callable[[str, str, object], tuple[dict[str, Any], Any | None]]


class ModelProvider:
    def stream_turn(
        self,
        *,
        context: ProviderContext,
        tool_dispatch: ToolDispatch,
    ) -> Iterator[ProviderEvent]:
        raise NotImplementedError
