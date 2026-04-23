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
    r = client.get("/me/documents")
    assert r.status_code == 200
    assert r.json() == {"items": []}


@pytest.mark.api
def test_upload_rejects_empty_file(client: TestClient, dry_api_env: None) -> None:
    r = client.post(
        "/me/documents/upload",
        files={"file": ("empty.txt", BytesIO(b""), "text/plain")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.upload_user_document")
def test_upload_success_mocked(
    upload: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    upload.return_value = ("documents/u1/report.pdf", "gs://bucket/documents/u1/report.pdf")
    r = client.post(
        "/me/documents/upload",
        files={"file": ("report.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["file_name"] == "report.pdf"
    assert data["storage_path"].endswith("report.pdf")


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
    r = client.get("/me/documents")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["file_name"] == "a.pdf"


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.analyze_uploaded_document")
@patch("fambot_backend.api.routers.documents.upload_user_document")
def test_analyze_uploaded_document_mocked(
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
        "/me/documents/analyze",
        files={"file": ("lab.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["file_name"] == "lab.pdf"
    assert data["storage_path"].endswith("lab.pdf")
    assert data["analysis_model"] == "gemini-test"
    assert "Stay active" in data["recommendations_text"]
    analyze.assert_called_once()
    kw = analyze.call_args.kwargs
    assert kw["uid"] == "dev-user"
    assert kw["file_name"] == "lab.pdf"
    assert kw["payload"] == b"%PDF-1.4"


@pytest.mark.api
@patch("fambot_backend.api.routers.documents.analyze_stored_document")
def test_analyze_stored_document_compat_mocked(
    analyze: object,
    client: TestClient,
    dry_api_env: None,
) -> None:
    analyze.return_value = {"model": "gemini-x", "analysis": "Summary text."}
    r = client.post("/documents/doc-1/analyze")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["doc_id"] == "doc-1"
    assert data["model"] == "gemini-x"
    assert data["analysis"] == "Summary text."
    analyze.assert_called_once_with(uid="dev-user", doc_id="doc-1")
