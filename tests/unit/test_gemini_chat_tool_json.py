"""Unit tests for JSON helpers used in Gemini chat tools."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from fambot_backend.services import gemini_document_analysis as g


@pytest.mark.unit
def test_list_stored_documents_json() -> None:
    items = [
        {
            "file_name": "a.pdf",
            "size_bytes": 10,
            "content_type": "application/pdf",
            "updated_at": None,
        }
    ]
    with patch.object(g, "list_user_documents", return_value=items):
        s = g._list_stored_documents_json("u1")
    d = json.loads(s)
    assert d["count"] == 1
    assert d["stored_documents"][0]["file_name"] == "a.pdf"


@pytest.mark.unit
def test_family_lifestyle_risk_json_empty() -> None:
    with patch.object(g, "family_peers_for_scoring", return_value=[]):
        s = g._family_lifestyle_risk_json("u1")
    d = json.loads(s)
    assert d["family_members"] == []
