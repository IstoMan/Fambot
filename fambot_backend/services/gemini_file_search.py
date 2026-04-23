"""Gemini File Search store lifecycle: per-user store + ingest from Firebase Storage uploads."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from fambot_backend.services.firestore_users import get_file_search_store_name, set_file_search_store_name


def _get_client() -> Any:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY required")
    from google import genai

    return genai.Client(api_key=api_key)


def file_search_disabled() -> bool:
    return os.environ.get("FAMBOT_GEMINI_DISABLE_FILE_SEARCH", "").strip() == "1"


def _poll_operation(client: Any, op: Any, *, timeout_s: float = 120.0) -> Any:
    start = time.monotonic()
    current = op
    while not getattr(current, "done", False):
        if time.monotonic() - start > timeout_s:
            raise HTTPException(status_code=504, detail="File Search indexing timed out")
        time.sleep(0.7)
        current = client.operations.get(current)
    err = getattr(current, "error", None)
    if err:
        raise HTTPException(status_code=502, detail=f"File Search operation failed: {err}")
    return current


def get_or_create_file_search_store(uid: str) -> str | None:
    """Return Gemini file search store resource name, creating one if missing."""
    if os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1":
        return None
    if file_search_disabled():
        return None
    existing = get_file_search_store_name(uid)
    if existing:
        return existing
    client = _get_client()
    from google.genai import types as genai_types

    try:
        store = client.file_search_stores.create(
            config=genai_types.CreateFileSearchStoreConfig(
                display_name=f"fambot-{uid[:24]}",
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"File Search store create failed: {exc}") from exc
    name = getattr(store, "name", None)
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=502, detail="File Search store missing name")
    set_file_search_store_name(uid, name)
    return name


def ingest_bytes_to_file_search(uid: str, *, file_name: str, content_type: str, payload: bytes) -> None:
    """Upload bytes to the user's File Search store (blocking until indexed)."""
    if os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1":
        return
    if file_search_disabled() or not payload:
        return
    store = get_or_create_file_search_store(uid)
    if not store:
        return
    client = _get_client()
    suffix = Path(file_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        path = tmp.name
    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store,
            file=path,
            config={"mime_type": content_type or "application/octet-stream"},
        )
        _poll_operation(client, op)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"File Search ingest failed: {exc}") from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


