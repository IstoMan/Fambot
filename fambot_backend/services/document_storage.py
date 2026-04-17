from __future__ import annotations

import os
import uuid
from pathlib import Path

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

    suffix = Path(upload.filename or "document").suffix
    object_name = f"documents/{uid}/{uuid.uuid4().hex}{suffix}"
    blob = bucket.blob(object_name)

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    content_type = upload.content_type or "application/octet-stream"
    blob.upload_from_string(payload, content_type=content_type)

    if os.environ.get("FAMBOT_STORAGE_MAKE_PUBLIC") == "1":
        blob.make_public()

    return object_name, f"gs://{bucket.name}/{object_name}"
