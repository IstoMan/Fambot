from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    DocumentSearchIn,
    DocumentSearchOut,
    DocumentUploadOut,
    UserDocumentsListOut,
)
from fambot_backend.services.document_storage import (
    get_user_document_payload,
    list_user_documents,
    upload_user_document,
)
from fambot_backend.services.firestore_users import get_user_profile

router = APIRouter(prefix="/me/documents", tags=["me", "documents"])


@router.post("/upload", response_model=DocumentUploadOut)
def upload_document(
    file: UploadFile = File(...),
    uid: str = Depends(firebase_uid),
) -> DocumentUploadOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    content_type = file.content_type or "application/octet-stream"
    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    storage_path, storage_uri = upload_user_document(uid, file, payload)
    
    return DocumentUploadOut(
        file_name=file.filename,
        content_type=content_type,
        storage_path=storage_path,
        storage_uri=storage_uri,
    )




@router.get("", response_model=UserDocumentsListOut)
def get_my_documents(uid: str = Depends(firebase_uid)) -> UserDocumentsListOut:
    return UserDocumentsListOut(items=list_user_documents(uid))


