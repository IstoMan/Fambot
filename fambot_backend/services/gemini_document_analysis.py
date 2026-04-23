from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent
from typing import Any

from fastapi import HTTPException

from fambot_backend.services.document_storage import (
    get_user_document,
    get_user_document_payload,
    list_user_documents,
)
from fambot_backend.services.firestore_users import get_user_profile

CHAT_ASSISTANT_FALLBACK = (
    "I wasn't able to generate a response just now. Please try again in a moment."
)


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY required")
    from google import genai

    return genai.Client(api_key=api_key)


def _upload_bytes(client: Any, *, file_name: str, content_type: str, payload: bytes) -> Any:
    suffix = Path(file_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        temp_name = tmp.name
    try:
        return client.files.upload(file=temp_name, config={"mime_type": content_type})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini file upload failed: {exc}") from exc
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def _profile_context(uid: str) -> str:
    profile = get_user_profile(uid)
    profile_data = profile.model_dump(mode="json", exclude_none=True)
    lines = [f"- {k}: {v}" for k, v in sorted(profile_data.items()) if v is not None]
    return "\n".join(lines) if lines else "(No profile data)"


def _model_name() -> str:
    return os.environ.get("GEMINI_REPORT_MODEL", "gemini-2.5-flash")


def analyze_uploaded_document(
    *,
    uid: str,
    file_name: str,
    content_type: str,
    payload: bytes,
) -> dict[str, str]:
    client = _get_client()
    model_name = _model_name()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    uploaded = _upload_bytes(
        client,
        file_name=file_name,
        content_type=content_type or "application/octet-stream",
        payload=payload,
    )
    context_block = _profile_context(uid)
    prompt = dedent(
        """
        Analyze this medical document and provide practical prevention and lifestyle guidance.
        Focus on clear, non-alarmist language and mention when medical follow-up is recommended.
        End with a brief disclaimer that this is not a diagnosis.
        """
    ).strip()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                uploaded,
                f"USER PROFILE:\n{context_block}\n\n{prompt}",
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Document analysis failed: {exc}") from exc
    analysis = (response.text or "").strip()
    if not analysis:
        raise HTTPException(status_code=502, detail="Gemini returned empty analysis")
    return {"model": model_name, "analysis": analysis}


def analyze_stored_document(*, uid: str, doc_id: str) -> dict[str, str]:
    doc = get_user_document(uid, doc_id)
    storage_path = doc.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise HTTPException(status_code=500, detail="Document storage path missing")
    payload = get_user_document_payload(storage_path)
    return analyze_uploaded_document(
        uid=uid,
        file_name=str(doc.get("filename") or doc_id),
        content_type=str(doc.get("content_type") or "application/octet-stream"),
        payload=payload,
    )


def _prepare_chat_turn(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    upload_name: str | None = None,
    upload_content_type: str | None = None,
    upload_payload: bytes | None = None,
) -> tuple[Any, str, list[Any]]:
    client = _get_client()
    model_name = _model_name()
    docs = list_user_documents(uid)[:3]
    context_block = _profile_context(uid)
    history = history or []

    gemini_parts: list[Any] = []
    for item in docs:
        storage_path = item.get("storage_path")
        if not isinstance(storage_path, str) or not storage_path:
            continue
        try:
            payload = get_user_document_payload(storage_path)
            gemini_parts.append(
                _upload_bytes(
                    client,
                    file_name=str(item.get("file_name") or "document"),
                    content_type=str(item.get("content_type") or "application/octet-stream"),
                    payload=payload,
                )
            )
        except HTTPException:
            continue

    if upload_payload:
        gemini_parts.append(
            _upload_bytes(
                client,
                file_name=upload_name or "attachment.bin",
                content_type=upload_content_type or "application/octet-stream",
                payload=upload_payload,
            )
        )

    transcript_lines: list[str] = []
    for row in history[-20:]:
        role = str(row.get("role") or "user")
        content = str(row.get("content") or "").strip()
        if content:
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(No history)"

    prompt = dedent(
        f"""
        You are Fambot, a health assistant.
        Use uploaded documents as primary evidence when available.
        Be concise, professional, and include a brief non-diagnostic disclaimer.

        USER PROFILE:
        {context_block}

        CHAT HISTORY:
        {transcript}

        USER MESSAGE:
        {user_message}
        """
    ).strip()
    return client, model_name, [*gemini_parts, prompt]


def maybe_new_chat_title(
    *, user_message: str, history: list[dict[str, Any]] | None = None
) -> str | None:
    history = history or []
    if any(str(row.get("role")) == "user" for row in history):
        return None
    client = _get_client()
    model_name = _model_name()
    try:
        title_response = client.models.generate_content(
            model=model_name,
            contents=(
                "Generate a short chat title (max 5 words, plain text only) for: "
                f"{user_message}"
            ),
        )
    except Exception:
        return None
    candidate = (title_response.text or "").strip().strip('"').strip("'")
    return candidate or None


def generate_chat_turn(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    upload_name: str | None = None,
    upload_content_type: str | None = None,
    upload_payload: bytes | None = None,
) -> dict[str, Any]:
    client, model_name, contents = _prepare_chat_turn(
        uid=uid,
        user_message=user_message,
        history=history,
        upload_name=upload_name,
        upload_content_type=upload_content_type,
        upload_payload=upload_payload,
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        text = CHAT_ASSISTANT_FALLBACK

    new_title = maybe_new_chat_title(user_message=user_message, history=history)
    return {
        "model": model_name,
        "content": text,
        "citations": None,
        "new_title": new_title,
    }


def generate_chat_turn_stream(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    upload_name: str | None = None,
    upload_content_type: str | None = None,
    upload_payload: bytes | None = None,
) -> Iterator[str]:
    client, model_name, contents = _prepare_chat_turn(
        uid=uid,
        user_message=user_message,
        history=history,
        upload_name=upload_name,
        upload_content_type=upload_content_type,
        upload_payload=upload_payload,
    )
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
        ):
            t = getattr(chunk, "text", None) or ""
            if t:
                yield t
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}") from exc


def chat_with_documents(uid: str, user_query: str | None = None) -> dict[str, str]:
    result = generate_chat_turn(uid=uid, user_message=user_query or "Summarize my health.", history=[])
    return {
        "recommendations_text": str(result.get("content") or "").strip(),
        "model": str(result.get("model") or _model_name()),
        "query_used": user_query or "General Summary",
    }
