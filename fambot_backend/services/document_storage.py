from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from firebase_admin import storage

from fambot_backend.core.firebase_init import init_firebase


def upload_user_document(uid: str, upload: UploadFile, payload: bytes) -> tuple[str, str]:
    """Upload a user document to Firebase Storage and return (blob_path, gs_uri)."""
    init_firebase()
    bucket = storage.bucket()
    if not bucket.name:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: FIREBASE_STORAGE_BUCKET is required",
        )

    # Use the original filename to prevent duplicates (overwrites if same name exists)
    safe_filename = Path(upload.filename or "document").name
    object_name = f"documents/{uid}/{safe_filename}"
    blob = bucket.blob(object_name)
    
    # Check if a file with the same name already exists to avoid listing confusion
    # though upload_from_string will naturally overwrite it.

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    content_type = upload.content_type or "application/octet-stream"
    blob.upload_from_string(payload, content_type=content_type)

    if os.environ.get("FAMBOT_STORAGE_MAKE_PUBLIC") == "1":
        blob.make_public()

    return object_name, f"gs://{bucket.name}/{object_name}"


def list_user_documents(uid: str) -> list[dict[str, Any]]:
    """List documents for one user from Firebase Storage prefix."""
    init_firebase()
    bucket = storage.bucket()
    if not bucket.name:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: FIREBASE_STORAGE_BUCKET is required",
        )

    prefix = f"documents/{uid}/"
    items: list[dict[str, Any]] = []
    for blob in bucket.list_blobs(prefix=prefix):
        updated_at = blob.updated
        if isinstance(updated_at, datetime) and updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=None)
        items.append(
            {
                "file_name": Path(blob.name).name,
                "content_type": blob.content_type or "application/octet-stream",
                "storage_path": blob.name,
                "storage_uri": f"gs://{bucket.name}/{blob.name}",
                "size_bytes": blob.size or 0,
                "updated_at": updated_at,
            }
        )
    def _sort_key(item: dict[str, Any]) -> str:
        updated = item.get("updated_at")
        if isinstance(updated, datetime):
            return updated.isoformat()
        return ""

    items.sort(key=_sort_key, reverse=True)
    return items


def get_user_document_payload(storage_path: str) -> bytes:
    """Download the actual file bytes for a document from storage."""
    init_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Document not found in storage")
    return blob.download_as_string()


def get_user_document(uid: str, doc_id: str) -> dict[str, Any]:
    """Resolve one user document by id (filename)."""
    for item in list_user_documents(uid):
        if item.get("file_name") == doc_id:
            created = item.get("updated_at")
            return {
                "id": doc_id,
                "filename": doc_id,
                "content_type": item.get("content_type") or "application/octet-stream",
                "size": int(item.get("size_bytes") or 0),
                "storage_path": item.get("storage_path"),
                "created_at": created,
            }
    raise HTTPException(status_code=404, detail="Document not found")


def delete_user_document(uid: str, doc_id: str) -> None:
    """Delete one user document by id (filename)."""
    item = get_user_document(uid, doc_id)
    storage_path = item.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise HTTPException(status_code=500, detail="Document storage path missing")
    init_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Document not found in storage")
    blob.delete()
