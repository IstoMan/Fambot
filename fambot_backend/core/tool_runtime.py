from __future__ import annotations

from typing import Any

from fambot_backend.providers.gemini_provider import tool_dispatch_to_provider_payload
from fambot_backend.services.gemini_document_analysis import _tool_dispatch


def dispatch_tool(
    uid: str,
    name: str,
    args: object,
) -> tuple[dict[str, Any], Any | None]:
    return tool_dispatch_to_provider_payload(
        uid=uid,
        name=name,
        args=args,
        dispatch_fn=_tool_dispatch,
    )
