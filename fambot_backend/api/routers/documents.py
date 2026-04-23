from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    DocumentAnalyzeOut,
    DocumentAnalyzeResponse,
    DocumentType,
    DocumentUploadOut,
    FeverDocumentResponse,
    UserDocumentsListOut,
)
from fambot_backend.services.document_storage import (
    delete_user_document,
    get_user_document,
    list_user_documents,
    upload_user_document,
)
from fambot_backend.services.gemini_document_analysis import (
    analyze_stored_document,
    analyze_uploaded_document,
)
from fambot_backend.services.gemini_file_search import ingest_bytes_to_file_search

router = APIRouter(tags=["documents"])


@router.post("/me/documents/upload", response_model=DocumentUploadOut)
def upload_my_document(
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
    try:
        ingest_bytes_to_file_search(
            uid,
            file_name=file.filename,
            content_type=content_type,
            payload=payload,
        )
    except Exception:
        pass
    return DocumentUploadOut(
        file_name=file.filename,
        content_type=content_type,
        storage_path=storage_path,
        storage_uri=storage_uri,
    )


@router.get("/me/documents", response_model=UserDocumentsListOut)
def get_my_documents(uid: str = Depends(firebase_uid)) -> UserDocumentsListOut:
    return UserDocumentsListOut(items=list_user_documents(uid))


@router.post("/me/documents/analyze", response_model=DocumentAnalyzeOut)
def analyze_my_uploaded_document(
    file: UploadFile = File(...),
    uid: str = Depends(firebase_uid),
) -> DocumentAnalyzeOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")
    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    content_type = file.content_type or "application/octet-stream"
    storage_path, storage_uri = upload_user_document(uid, file, payload)
    try:
        ingest_bytes_to_file_search(
            uid,
            file_name=file.filename,
            content_type=content_type,
            payload=payload,
        )
    except Exception:
        pass
    result = analyze_uploaded_document(
        uid=uid,
        file_name=file.filename,
        content_type=content_type,
        payload=payload,
    )
    return DocumentAnalyzeOut(
        file_name=file.filename,
        content_type=content_type,
        storage_path=storage_path,
        storage_uri=storage_uri,
        analysis_model=str(result.get("model") or ""),
        recommendations_text=str(result.get("analysis") or ""),
    )


@router.post("/documents/upload", response_model=FeverDocumentResponse)
def upload_document_compat(
    file: UploadFile = File(...),
    type: DocumentType | None = Form(None),
    uid: str = Depends(firebase_uid),
) -> FeverDocumentResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")
    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    storage_path, _storage_uri = upload_user_document(uid, file, payload)
    try:
        ingest_bytes_to_file_search(
            uid,
            file_name=file.filename,
            content_type=file.content_type or "application/octet-stream",
            payload=payload,
        )
    except Exception:
        pass
    now = datetime.now(timezone.utc)
    return FeverDocumentResponse(
        id=file.filename,
        filename=file.filename,
        type=type,
        content_type=file.content_type or "application/octet-stream",
        size=len(payload),
        storage_path=storage_path,
        created_at=now,
    )


@router.get("/documents", response_model=list[FeverDocumentResponse])
def list_documents_compat(uid: str = Depends(firebase_uid)) -> list[FeverDocumentResponse]:
    items = list_user_documents(uid)
    out: list[FeverDocumentResponse] = []
    now = datetime.now(timezone.utc)
    for item in items:
        created_at = item.get("updated_at")
        out.append(
            FeverDocumentResponse(
                id=str(item.get("file_name") or ""),
                filename=str(item.get("file_name") or ""),
                type=None,
                content_type=str(item.get("content_type") or "application/octet-stream"),
                size=int(item.get("size_bytes") or 0),
                storage_path=str(item.get("storage_path") or ""),
                created_at=created_at if isinstance(created_at, datetime) else now,
            )
        )
    return out


@router.get("/documents/{doc_id}", response_model=FeverDocumentResponse)
def get_document_compat(doc_id: str, uid: str = Depends(firebase_uid)) -> FeverDocumentResponse:
    item = get_user_document(uid, doc_id)
    created_at = item.get("created_at")
    return FeverDocumentResponse(
        id=str(item.get("id") or doc_id),
        filename=str(item.get("filename") or doc_id),
        type=None,
        content_type=str(item.get("content_type") or "application/octet-stream"),
        size=int(item.get("size") or 0),
        storage_path=str(item.get("storage_path") or ""),
        created_at=created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc),
    )


@router.post("/documents/{doc_id}/analyze", response_model=DocumentAnalyzeResponse)
def analyze_document_compat(doc_id: str, uid: str = Depends(firebase_uid)) -> DocumentAnalyzeResponse:
    result = analyze_stored_document(uid=uid, doc_id=doc_id)
    return DocumentAnalyzeResponse(
        doc_id=doc_id,
        model=str(result.get("model") or ""),
        analysis=str(result.get("analysis") or ""),
    )


@router.delete("/documents/{doc_id}")
def delete_document_compat(doc_id: str, uid: str = Depends(firebase_uid)) -> dict[str, str]:
    delete_user_document(uid, doc_id)
    return {"status": "success", "detail": "Document deleted"}


