"""Document routes with storage layer mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.list_user_documents")
def test_list_documents_empty(
    list_docs: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    list_docs.return_value = []
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.api
def test_upload_rejects_empty_file(client: TestClient, dry_api_env: None) -> None:
    r = client.post(
        "/documents",
        files={"file": ("empty.txt", BytesIO(b""), "text/plain")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.upload_user_document")
@patch("fambot_backend.api.routers.documents.ingest_bytes_to_file_search")
def test_upload_success_mocked(
    _ingest: object,
    upload: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    upload.return_value = ("documents/u1/report.pdf", "gs://bucket/documents/u1/report.pdf")
    r = client.post(
        "/documents",
        files={"file": ("report.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["filename"] == "report.pdf"
    assert data["id"] == "report.pdf"
    assert data["storage_path"].endswith("report.pdf")
    assert data["analysis_model"] is None
    assert data["recommendations_text"] is None


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.list_user_documents")
def test_list_documents_returns_items(
    list_docs: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    now = datetime.now(timezone.utc)
    list_docs.return_value = [
        {
            "file_name": "a.pdf",
            "content_type": "application/pdf",
            "storage_path": "documents/u/a.pdf",
            "storage_uri": "gs://b/documents/u/a.pdf",
            "size_bytes": 12,
            "updated_at": now,
        }
    ]
    r = client.get("/documents")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["filename"] == "a.pdf"
    assert items[0]["id"] == "a.pdf"
    assert items[0]["size_bytes"] == 12


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.analyze_uploaded_document")
@patch("fambot_backend.api.routers.documents.upload_user_document")
@patch("fambot_backend.api.routers.documents.ingest_bytes_to_file_search")
def test_upload_with_analyze_mocked(
    _ingest: object,
    upload: object,
    analyze: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    upload.return_value = ("documents/dev-user/lab.pdf", "gs://bucket/documents/dev-user/lab.pdf")
    analyze.return_value = {
        "model": "gemini-test",
        "analysis": "Stay active. Not a diagnosis.",
    }
    r = client.post(
        "/documents",
        files={"file": ("lab.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
        data={"analyze": "true"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["filename"] == "lab.pdf"
    assert data["analysis_model"] == "gemini-test"
    assert "Stay active" in (data["recommendations_text"] or "")
    analyze.assert_called_once()
    kw = analyze.call_args.kwargs
    assert kw["uid"] == "dev-user"
    assert kw["file_name"] == "lab.pdf"
    assert kw["payload"] == b"%PDF-1.4"


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.get_user_document_payload")
@patch("fambot_backend.api.routers.documents.get_user_document")
def test_download_document_mocked(
    get_doc: object,
    get_payload: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    get_doc.return_value = {
        "id": "report.pdf",
        "filename": "report.pdf",
        "content_type": "application/pdf",
        "size": 4,
        "storage_path": "documents/dev-user/report.pdf",
        "storage_uri": "gs://b/documents/dev-user/report.pdf",
        "created_at": None,
    }
    get_payload.return_value = b"%PDF"
    r = client.get("/documents/report.pdf/download")
    assert r.status_code == 200, r.text
    assert r.content == b"%PDF"
    assert r.headers.get("content-type", "").startswith("application/pdf")
    assert "attachment" in (r.headers.get("content-disposition") or "")
    assert "report.pdf" in (r.headers.get("content-disposition") or "")
    get_payload.assert_called_once_with("documents/dev-user/report.pdf")


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.analyze_stored_document")
def test_analyze_stored_document_mocked(
    analyze: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    analyze.return_value = {"model": "gemini-x", "analysis": "Summary text."}
    r = client.post("/documents/doc-1/analyze")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["doc_id"] == "doc-1"
    assert data["analysis_model"] == "gemini-x"
    assert data["recommendations_text"] == "Summary text."
    analyze.assert_called_once_with(uid="dev-user", doc_id="doc-1")
