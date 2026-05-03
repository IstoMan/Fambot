from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import DocumentAnalysisResult, DocumentItem, DocumentType
from fambot_backend.services.document_storage import (
    delete_user_document,
    get_user_document,
    get_user_document_payload,
    list_user_documents,
    upload_user_document,
)
from fambot_backend.services.gemini_document_analysis import (
    analyze_stored_document,
    analyze_uploaded_document,
)
from fambot_backend.services.gemini_file_search import ingest_bytes_to_file_search

router = APIRouter(tags=["documents"])


def _attachment_content_disposition(filename: str) -> str:
    """RFC 5987 Content-Disposition for UTF-8 filenames."""
    safe = filename.replace("\\", "\\\\").replace('"', '\\"')
    try:
        safe.encode("ascii")
        return f'attachment; filename="{safe}"'
    except UnicodeEncodeError:
        ascii_fallback = (filename.encode("ascii", "ignore").decode("ascii") or "download").strip() or "download"
        if not Path(ascii_fallback).suffix and (suf := Path(filename).suffix):
            ascii_fallback = ascii_fallback + suf
        encoded = quote(filename, safe="")
        return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


def _document_from_list_row(row: dict) -> DocumentItem:
    updated = row.get("updated_at")
    return DocumentItem(
        id=str(row.get("file_name") or ""),
        filename=str(row.get("file_name") or ""),
        content_type=str(row.get("content_type") or "application/octet-stream"),
        size_bytes=int(row.get("size_bytes") or 0),
        storage_path=str(row.get("storage_path") or ""),
        storage_uri=str(row["storage_uri"]) if row.get("storage_uri") else None,
        updated_at=updated if isinstance(updated, datetime) else None,
        type=None,
    )


def _document_from_get_row(row: dict) -> DocumentItem:
    updated = row.get("created_at")
    storage_path = row.get("storage_path")
    return DocumentItem(
        id=str(row.get("id") or ""),
        filename=str(row.get("filename") or ""),
        content_type=str(row.get("content_type") or "application/octet-stream"),
        size_bytes=int(row.get("size") or 0),
        storage_path=str(storage_path or ""),
        storage_uri=str(row["storage_uri"]) if row.get("storage_uri") else None,
        updated_at=updated if isinstance(updated, datetime) else None,
        type=None,
    )


@router.get("/documents", response_model=list[DocumentItem])
def list_documents(uid: str = Depends(firebase_uid)) -> list[DocumentItem]:
    return [_document_from_list_row(item) for item in list_user_documents(uid)]


@router.post("/documents", response_model=DocumentItem)
def upload_document(
    file: UploadFile = File(...),
    type: DocumentType | None = Form(None),
    analyze: bool = Form(False),
    uid: str = Depends(firebase_uid),
) -> DocumentItem:
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
    now = datetime.now(timezone.utc)
    analysis_model: str | None = None
    recommendations_text: str | None = None
    if analyze:
        result = analyze_uploaded_document(
            uid=uid,
            file_name=file.filename,
            content_type=content_type,
            payload=payload,
        )
        analysis_model = str(result.get("model") or "")
        recommendations_text = str(result.get("analysis") or "")
    return DocumentItem(
        id=file.filename,
        filename=file.filename,
        content_type=content_type,
        size_bytes=len(payload),
        storage_path=storage_path,
        storage_uri=storage_uri,
        updated_at=now,
        type=type,
        analysis_model=analysis_model,
        recommendations_text=recommendations_text,
    )


@router.get("/documents/{doc_id}", response_model=DocumentItem)
def get_document(doc_id: str, uid: str = Depends(firebase_uid)) -> DocumentItem:
    return _document_from_get_row(get_user_document(uid, doc_id))


@router.get("/documents/{doc_id}/download")
def download_document(doc_id: str, uid: str = Depends(firebase_uid)) -> Response:
    """Stream the raw file bytes so the client can open or save the uploaded document."""
    doc = get_user_document(uid, doc_id)
    storage_path = doc.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise HTTPException(status_code=500, detail="Document storage path missing")
    payload = get_user_document_payload(storage_path)
    media_type = str(doc.get("content_type") or "application/octet-stream")
    filename = str(doc.get("filename") or doc_id)
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": _attachment_content_disposition(filename)},
    )


@router.post("/documents/{doc_id}/analyze", response_model=DocumentAnalysisResult)
def analyze_document(doc_id: str, uid: str = Depends(firebase_uid)) -> DocumentAnalysisResult:
    result = analyze_stored_document(uid=uid, doc_id=doc_id)
    return DocumentAnalysisResult(
        doc_id=doc_id,
        analysis_model=str(result.get("model") or ""),
        recommendations_text=str(result.get("analysis") or ""),
    )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, uid: str = Depends(firebase_uid)) -> dict[str, str]:
    delete_user_document(uid, doc_id)
    return {"status": "success", "detail": "Document deleted"}
