from __future__ import annotations

import os
import tempfile
from pathlib import Path
from textwrap import dedent

from fastapi import HTTPException


def analyze_report_with_gemini(
    *,
    filename: str,
    content_type: str,
    payload: bytes,
    clinical_context: str | None = None,
) -> dict[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: GEMINI_API_KEY is required",
        )

    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: install google-genai dependency",
        ) from exc

    model_name = os.environ.get("GEMINI_REPORT_MODEL", "gemini-2.5-flash")
    prompt = dedent(
        f"""
        You are a preventive-care assistant helping a user understand an uploaded health report.
        Analyze the file contents and answer in plain text with these exact section labels:
        1) Prevention Summary
        2) Lifestyle Recommendations
        3) Questions For Doctor
        4) Safety Note

        Guidance:
        - Keep language simple, direct, and non-alarmist.
        - Provide practical daily habits (diet, sleep, movement, stress, smoking/alcohol where relevant).
        - If the report lacks enough information, say what is missing.
        - Never claim diagnosis certainty.
        - Include a short disclaimer that this does not replace medical care.
        - If a metric looks urgent, explicitly advise immediate clinician follow-up.

        Optional user context:
        {clinical_context or "No additional context provided."}
        """
    ).strip()

    client = genai.Client(api_key=api_key)
    safe_name = Path(filename).name or "report.bin"
    suffix = Path(safe_name).suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(payload)
            tmp.flush()
            uploaded = client.files.upload(file=tmp.name, config={"mime_type": content_type})
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded, prompt],
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini analysis failed: {exc}") from exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty analysis for the uploaded document",
        )

    return {"analysis_text": raw_text, "model": model_name}
