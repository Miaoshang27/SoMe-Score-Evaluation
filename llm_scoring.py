import json, time, pathlib, random, os
from typing import Dict, Any

# ---- Config ----
INPUT = "posts_3.json"
OUT   = "posts_scored_3.jsonl"
TEMP  = 0.2
RATE_LIMIT_QPS = 1.2  # be gentle

# ---- Gemini setup ----
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("Set GOOGLE_API_KEY in your environment.")

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = (
    "You are a marketing quality rater for social media captions. "
    "Score the caption using the provided rubric. Output must be valid JSON only."
)

MODEL_NAME = "gemini-2.0-flash"  # or another available Gemini model

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
)

def build_user_prompt(caption: str) -> str:
    return f"""Context:
- Platform: Instagram (caption-only; no image/video available)
- Goal: Predict post quality that tends to drive conversion and follower gains.

Rubric (0–5 each, integers):
- clarity: clear, well-structured, concrete phrasing
- hook: strength of the first 1–2 sentences to grab attention
- value_delivery: actionable info, useful benefit, or clear offer
- cta: explicit, relevant, unobtrusive call to action
- brand_fit: tone & promise match a typical online fitness/coach brand
- hashtag_use: topical and not spammy (0 if absent; 5 = concise + on-topic)
- compliance: avoids risky/forbidden claims
- language_fit: grammar/spelling; if Danish/Swedish/English, evaluate appropriately
- engagement_prompt: asks a question, prompt to comment/share, etc.

Also return overall_quality (0–5) as your weighted summary (consistent across posts).

Caption:
{caption}

JSON Schema (enforce this exactly):
{{"type":"object","properties":{{"scores":{{"type":"object","properties":{{"clarity":{{"type":"integer","minimum":0,"maximum":5}},"hook":{{"type":"integer","minimum":0,"maximum":5}},"value_delivery":{{"type":"integer","minimum":0,"maximum":5}},"cta":{{"type":"integer","minimum":0,"maximum":5}},"brand_fit":{{"type":"integer","minimum":0,"maximum":5}},"hashtag_use":{{"type":"integer","minimum":0,"maximum":5}},"compliance":{{"type":"integer","minimum":0,"maximum":5}},"language_fit":{{"type":"integer","minimum":0,"maximum":5}},"engagement_prompt":{{"type":"integer","minimum":0,"maximum":5}}}},"required":["clarity","hook","value_delivery","cta","brand_fit","hashtag_use","compliance","language_fit","engagement_prompt"]}},"overall_quality":{{"type":"integer","minimum":0,"maximum":5}},"rationale_short":{{"type":"string","maxLength":200}}}},"required":["scores","overall_quality","rationale_short"],"additionalProperties":false}}"""

def llm_score(caption: str) -> Dict[str, Any]:
    """Call Gemini, enforce JSON, try a light repair if needed."""
    user_prompt = build_user_prompt(caption[:2000])
    for attempt in range(2):
        resp = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMP,
                response_mime_type="application/json",  # <-- ask for JSON only
            ),
        )
        txt = (resp.text or "").strip()
        try:
            obj = json.loads(txt)
            # quick shape checks
            assert "scores" in obj and "overall_quality" in obj
            return obj
        except Exception:
            if attempt == 0:
                # light repair attempt: trim to outermost braces
                start, end = txt.find("{"), txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    frag = txt[start : end + 1]
                    try:
                        obj = json.loads(frag)
                        assert "scores" in obj and "overall_quality" in obj
                        return obj
                    except Exception:
                        pass
            raise ValueError(f"Invalid JSON from model: {txt[:400]}")

def rate_limit_sleep(qps: float):
    time.sleep(1.0 / max(qps, 0.1) + random.uniform(0, 0.2))

# ---- Run ----
data = json.loads(pathlib.Path(INPUT).read_text(encoding="utf-8"))
with open(OUT, "w", encoding="utf-8") as outf:
    for i, post in enumerate(data, 1):
        cap = (post.get("caption") or "").strip()
        if not cap:
            continue
        try:
            scores = llm_score(cap)
            result = {
                "url": post.get("url"),
                "ownerUsername": post.get("ownerUsername"),
                "timestamp": post.get("timestamp"),
                "caption": cap,
                "llm_scores": scores,
            }
        except Exception as e:
            result = {"url": post.get("url"), "error": str(e)}
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")

        if i % 25 == 0:
            print(f"Scored {i} posts...")
        rate_limit_sleep(RATE_LIMIT_QPS)

print(f"Done → {OUT}")
