from __future__ import annotations
from collections.abc import Iterator
from typing import Any

from fastapi import HTTPException

from fambot_backend.providers.model_provider import (
    ModelProvider,
    ProviderContext,
    ProviderEvent,
    ToolDispatch,
)
from fambot_backend.services.gemini_document_analysis import (
    CHAT_ASSISTANT_FALLBACK,
    CHAT_SYSTEM_INSTRUCTION,
    _build_tools_list,
    _citations_from_response,
    _fr_payload,
    _get_client,
    _model_name,
    _part_from_genai_upload,
    _upload_bytes,
    _user_message_text,
)


class GeminiProvider(ModelProvider):
    def stream_turn(
        self,
        *,
        context: ProviderContext,
        tool_dispatch: ToolDispatch,
    ) -> Iterator[ProviderEvent]:
        from google.genai import types

        client = _get_client()
        model_name = _model_name()
        tools = _build_tools_list(context.uid)
        user_text = _user_message_text(
            uid=context.uid,
            user_message=context.user_message,
            history=context.history,
        )
        user_parts: list[Any] = [types.Part(text=user_text)]
        if context.upload_payload:
            upload = _upload_bytes(
                client,
                file_name=context.upload_name or "attachment.bin",
                content_type=context.upload_content_type or "application/octet-stream",
                payload=context.upload_payload,
            )
            user_parts.append(_part_from_genai_upload(upload))

        contents: list[types.Content] = [types.Content(role="user", parts=user_parts)]
        config = types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for _ in range(max(1, context.max_tool_rounds)):
            token_seen = False
            parts_seen: list[Any] = []
            chunks_seen: list[Any] = []
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                chunks_seen.append(chunk)
                text_piece = (getattr(chunk, "text", None) or "").strip()
                if text_piece:
                    token_seen = True
                    yield ProviderEvent(kind="token", text=text_piece, model=model_name)
                parts_seen.extend(_parts_from_response_like(chunk))

            tool_call = _first_function_call(parts_seen)
            if tool_call is None:
                if token_seen:
                    # The streamed text is already emitted; include final metadata for citations.
                    citations = _citations_from_chunks(chunks_seen)
                    yield ProviderEvent(kind="done", model=model_name, citations=citations)
                    return
                # If streaming yielded no token text, fallback to one-shot output.
                text = _fallback_text_from_chunks(chunks_seen) or CHAT_ASSISTANT_FALLBACK
                yield ProviderEvent(kind="token", text=text, model=model_name)
                yield ProviderEvent(
                    kind="done",
                    model=model_name,
                    citations=_citations_from_chunks(chunks_seen),
                )
                return

            if parts_seen:
                contents.append(types.Content(role="model", parts=list(parts_seen)))

            name = str(getattr(tool_call, "name", "") or "")
            args = getattr(tool_call, "args", None)
            args_dict = args if isinstance(args, dict) else {}
            call_id = str(getattr(tool_call, "id", None) or "call")
            yield ProviderEvent(
                kind="tool_call",
                tool_name=name,
                tool_args=args_dict,
                tool_call_id=call_id,
                model=model_name,
            )
            payload, file_ref = tool_dispatch(context.uid, name, args_dict)
            yield ProviderEvent(
                kind="tool_result",
                tool_name=name,
                tool_args=payload,
                tool_call_id=call_id,
                model=model_name,
            )
            function_response = types.FunctionResponse(
                id=call_id,
                name=name,
                response=payload,
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=function_response)],
                )
            )
            if file_ref is not None and name == "include_stored_document":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            _part_from_genai_upload(file_ref),
                            types.Part(
                                text=(
                                    "Incorporate this file when replying. Summarize only relevant details "
                                    "and do not invent values not present in the file."
                                )
                            ),
                        ],
                    )
                )

        raise HTTPException(status_code=502, detail="Chat stopped after max tool rounds")


def _parts_from_response_like(response_like: Any) -> list[Any]:
    candidates = getattr(response_like, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    return list(parts or [])


def _first_function_call(parts: list[Any]) -> Any | None:
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None and getattr(function_call, "name", None):
            return function_call
    return None


def _citations_from_chunks(chunks: list[Any]) -> list[dict[str, Any]] | None:
    for chunk in reversed(chunks):
        citations = _citations_from_response(chunk)
        if citations:
            return citations
    return None


def _fallback_text_from_chunks(chunks: list[Any]) -> str:
    pieces: list[str] = []
    for chunk in chunks:
        text_piece = getattr(chunk, "text", None)
        if isinstance(text_piece, str) and text_piece:
            pieces.append(text_piece)
    return "".join(pieces).strip()


def tool_dispatch_to_provider_payload(
    *,
    uid: str,
    name: str,
    args: object,
    dispatch_fn: Any,
) -> tuple[dict[str, Any], Any | None]:
    result = dispatch_fn(uid, name, args)
    payload = _fr_payload(result)
    return payload, result.file_ref
