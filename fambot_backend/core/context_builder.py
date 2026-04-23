from __future__ import annotations

from typing import Any


def build_chat_history_context(
    *,
    messages: list[dict[str, Any]],
    max_messages: int = 20,
) -> list[dict[str, Any]]:
    if max_messages <= 0:
        return []
    return list(messages[-max_messages:])
