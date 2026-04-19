from __future__ import annotations

import os
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

from fastapi import HTTPException


from fambot_backend.services.document_storage import (
    get_user_document_payload,
    list_user_documents,
)





def chat_with_documents(uid: str, user_query: str | None = None) -> dict[str, str]:
    """
    Fetch user reports from storage, upload to Gemini ephemeral API, and chat with them.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY required")

    from google import genai
    client = genai.Client(api_key=api_key)
    
    # Using the model name confirmed from your terminal list
    model_name = os.environ.get("GEMINI_REPORT_MODEL", "gemini-2.5-flash")

    # 1. Get recent documents (limit to top 3 for speed/token reasons)
    docs = list_user_documents(uid)[:3]
    
    # 2. Prepare Gemini files
    gemini_files = []
    temp_files = []
    try:
        for d in docs:
            payload = get_user_document_payload(d["storage_path"])
            suffix = Path(d["file_name"]).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(payload)
                tmp.flush()
                temp_files.append(tmp.name)
                
                uploaded = client.files.upload(
                    file=tmp.name, 
                    config={"mime_type": d["content_type"]}
                )
                gemini_files.append(uploaded)

        # 3. Build prompt
        query = user_query or "Summarize my health based on these reports."
        
        # User profile context (imported here to avoid circulars if any, but it's safe)
        from fambot_backend.services.firestore_users import get_user_profile
        profile = get_user_profile(uid)
        profile_data = profile.model_dump(mode="json", exclude_none=True)
        context_lines = [f"- {k}: {v}" for k, v in sorted(profile_data.items()) if v]
        context_block = "\n".join(context_lines) if context_lines else "(No profile data)"

        prompt = dedent(
            f"""
            You are a Fambot Health Assistant. You have access to the user's uploaded medical reports and their profile.
            
            USER QUERY: {query}
            
            USER PROFILE:
            {context_block}
            
            GOAL:
            - Answer the query using the documents as the primary source.
            - Provide clear prevention and lifestyle recommendations.
            - Be professional, non-alarmist, and concise.
            - End with a medical advice disclaimer.
            """
        ).strip()

        # 4. Generate
        contents = gemini_files + [prompt]
        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}")
    finally:
        # Cleanup temp files
        for tf in temp_files:
            try:
                os.unlink(tf)
            except:
                pass

    return {
        "recommendations_text": (response.text or "").strip(),
        "model": model_name,
        "query_used": query or "General Summary"
    }
