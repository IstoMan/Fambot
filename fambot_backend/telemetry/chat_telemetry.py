from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

logger = logging.getLogger("fambot.chat")


def start_turn_trace(*, chat_id: str, turn_id: str, uid: str) -> float:
    logger.info(
        "chat.turn.start",
        extra={"chat_id": chat_id, "turn_id": turn_id, "uid": uid},
    )
    return perf_counter()


def finish_turn_trace(
    *,
    chat_id: str,
    turn_id: str,
    uid: str,
    state: str,
    started_at: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "turn_id": turn_id,
        "uid": uid,
        "state": state,
        "elapsed_ms": int((perf_counter() - started_at) * 1000),
    }
    if extra:
        payload.update(extra)
    logger.info("chat.turn.finish", extra=payload)
