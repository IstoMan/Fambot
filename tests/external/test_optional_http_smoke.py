"""Optional HTTP smoke against a URL you configure (default: skipped)."""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.external


def _external_enabled() -> bool:
    return os.environ.get("FAMBOT_RUN_EXTERNAL_TESTS") == "1"


@pytest.mark.skipif(not _external_enabled(), reason="Set FAMBOT_RUN_EXTERNAL_TESTS=1 to run external tests")
def test_get_external_health_url_if_set() -> None:
    url = os.environ.get("FAMBOT_EXTERNAL_HEALTH_URL", "").strip()
    if not url:
        pytest.skip("Set FAMBOT_EXTERNAL_HEALTH_URL to run this smoke test")
    r = httpx.get(url, timeout=15.0)
    assert r.status_code < 500
