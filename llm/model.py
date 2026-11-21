import os
import re
import json
import time
import warnings
from dotenv import load_dotenv

with warnings.catch_warnings():
    warnings.simplefilter("ignore", (FutureWarning, DeprecationWarning))
    import google.generativeai as genai

from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

FALLBACK_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

_model_instances = {}

def get_genai_model(model_name: str = None):
    global _model_instances
    if not model_name:
        model_name = FALLBACK_MODELS[0]
    if model_name not in _model_instances:
        _model_instances[model_name] = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.1,    # Low temperature for strict factual determinism
                "top_p": 0.8,
                "top_k": 40
            }
        )
    return _model_instances[model_name]


def ask_llm(prompt: str, max_retries: int = 4) -> str:
    """
    Execute LLM call with retry, backoff, and model fallback if quota is exhausted.
    """
    for model_name in FALLBACK_MODELS:
        model = get_genai_model(model_name)
        delay = 8.0

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                return ""
            except ResourceExhausted as exc:
                wait_time = delay
                err_str = str(exc)
                time_match = re.search(r"seconds:\s*(\d+)", err_str)
                if time_match:
                    wait_time = max(float(time_match.group(1)) + 2.0, delay)

                # If long retry wait (daily quota), break to try next fallback model
                if "daily" in err_str.lower() or wait_time > 60:
                    print(f"  [Quota Exceeded for {model_name}] Switching to fallback model...")
                    break

                if attempt == max_retries - 1:
                    print(f"  [Rate Limit for {model_name}] Trying next model...")
                    break

                print(f"  [Rate Limit] Backing off for {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                delay *= 1.5

            except GoogleAPIError as ge:
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 1.5
            except Exception as e:
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 1.5

    raise RuntimeError("All Gemini API models exhausted or unavailable. Please check your API quota.")


def parse_structured_response(raw_text: str, retrieved_chunks: list[dict]) -> dict:
    """
    Parse LLM response into structured dict and validate cited chunk IDs.
    Any chunk ID cited by the LLM that was not in the retrieved set is rejected.
    """
    valid_chunk_map = {c["chunk_id"]: c for c in retrieved_chunks}
    
    # Clean markdown JSON formatting if present
    clean_json_str = raw_text.strip()
    if clean_json_str.startswith("```"):
        clean_json_str = re.sub(r"^```(?:json)?\s*", "", clean_json_str)
        clean_json_str = re.sub(r"\s*```$", "", clean_json_str)

    parsed = None
    try:
        parsed = json.loads(clean_json_str)
    except Exception:
        # Fallback regex search for JSON block
        json_match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except Exception:
                pass

    if not isinstance(parsed, dict):
        # Fallback if model returned unstructured text
        return {
            "answerable": True,
            "answer": raw_text.strip(),
            "cited_chunk_ids": [],
            "verified_sources": [],
            "conflict_detected": False,
            "conflict_explanation": ""
        }

    raw_cited_ids = parsed.get("cited_chunk_ids", [])
    if not isinstance(raw_cited_ids, list):
        raw_cited_ids = []

    # Strict Programmatic Citation Validation
    verified_ids = []
    verified_sources = []

    for cid in raw_cited_ids:
        cid_str = str(cid).strip()
        if cid_str in valid_chunk_map:
            verified_ids.append(cid_str)
            chunk_info = valid_chunk_map[cid_str]
            verified_sources.append({
                "chunk_id": cid_str,
                "filename": chunk_info.get("filename", "unknown"),
                "page": chunk_info.get("page", 1),
                "snippet": chunk_info.get("content", "")[:120] + "..."
            })

    return {
        "answerable": bool(parsed.get("answerable", True)),
        "answer": str(parsed.get("answer", "")).strip(),
        "cited_chunk_ids": verified_ids,
        "verified_sources": verified_sources,
        "conflict_detected": bool(parsed.get("conflict_detected", False)),
        "conflict_explanation": str(parsed.get("conflict_explanation", "")).strip()
    }
