from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import DocumentAnalysisOut
from fambot_backend.services.document_storage import upload_user_document
from fambot_backend.services.gemini_reports import analyze_report_with_gemini

router = APIRouter(prefix="/me/documents", tags=["me", "documents"])


@router.post("/analyze", response_model=DocumentAnalysisOut)
def upload_and_analyze_document(
    file: UploadFile = File(...),
    clinical_context: str | None = Form(default=None),
    uid: str = Depends(firebase_uid),
) -> DocumentAnalysisOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    content_type = file.content_type or "application/octet-stream"
    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    storage_path, storage_uri = upload_user_document(uid, file, payload)
    analysis = analyze_report_with_gemini(
        filename=file.filename,
        content_type=content_type,
        payload=payload,
        clinical_context=clinical_context,
    )
    return DocumentAnalysisOut(
        file_name=file.filename,
        content_type=content_type,
        storage_path=storage_path,
        storage_uri=storage_uri,
        analysis_model=analysis["model"],
        analysis_text=analysis["analysis_text"],
    )
