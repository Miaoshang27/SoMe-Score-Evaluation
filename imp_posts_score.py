# imp_posts_score.py
# -------------------
# Add more LLM post-quality scores into an existing dataset (append/update).
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ======== CONFIG ========
CSV_IN   = "dataset_with_llm_scores.csv"          # existing file with some scores
CSV_OUT  = "dataset_with_llm_scores_updated.csv"  # write to a new file
JSONL_IN = "posts_scored_3.jsonl"                 # new scored posts
HANDLE_COL = "instagram_handle"
HIGH_QUALITY_THRESHOLD = 4                        # overall_quality >= 4
OVERWRITE_EXISTING = False                        # set True to override existing values
# =======================

def normalize_handle(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("https://", "").replace("http://", "")
    s = s.replace("www.", "")
    s = s.replace("instagram.com/", "")
    s = s.split("?")[0].split("/")[0]
    if s.startswith("@"):
        s = s[1:]
    return s.lower()

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

# ---- Load your CSV ----
df = pd.read_csv(CSV_IN)

if HANDLE_COL not in df.columns:
    raise ValueError(f"'{HANDLE_COL}' not found in {CSV_IN} columns: {list(df.columns)}")

# Create normalized join key
df["_handle_norm"] = df[HANDLE_COL].apply(normalize_handle)

# ---- Load new scored posts (JSONL) ----
rows = []
with open(JSONL_IN, "r", encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

raw = pd.DataFrame(rows)

# Keep only rows that have llm_scores
if "llm_scores" not in raw.columns:
    raise ValueError("No 'llm_scores' field in JSONL rows.")
raw = raw[raw["llm_scores"].notna()].copy()
if raw.empty:
    raise ValueError("No valid llm_scores found in JSONL. Check your scoring output.")

# ---- Flatten scores ----
flat = pd.json_normalize(raw["llm_scores"])
flat.columns = [c.replace("scores.", "score_") for c in flat.columns]
scored = pd.concat([raw.drop(columns=["llm_scores"]), flat], axis=1)

# Normalize owner handle from JSONL
if "ownerUsername" not in scored.columns:
    raise ValueError("'ownerUsername' not found in posts_scored JSONL rows.")
scored["_handle_norm"] = scored["ownerUsername"].apply(normalize_handle)

# Deduplicate by URL if repeats
if "timestamp" in scored.columns:
    scored = scored.sort_values("timestamp")
scored = scored.drop_duplicates(subset=["url"], keep="last")

# Ensure numeric types for score columns
int_cols = [
    "score_clarity", "score_hook", "score_value_delivery", "score_cta",
    "score_brand_fit", "score_hashtag_use", "score_compliance",
    "score_language_fit", "score_engagement_prompt", "overall_quality",
]
for c in int_cols:
    if c in scored.columns:
        scored[c] = pd.to_numeric(scored[c], errors="coerce")

# Flag high-quality posts
scored["is_high_quality"] = (scored["overall_quality"] >= HIGH_QUALITY_THRESHOLD).astype("Int64")

# ---- Aggregate per handle ----
agg_kwargs = {f"{c}_mean": (c, "mean") for c in int_cols if c in scored.columns}
agg = (
    scored.groupby("_handle_norm")
    .agg(
        posts_scored=("url", "count"),
        high_quality_rate=("is_high_quality", "mean"),
        **agg_kwargs
    )
    .reset_index()
)

# ---- Merge into existing dataset ----
# Give new columns a _new suffix to avoid collisions
out = df.merge(agg, on="_handle_norm", how="left", suffixes=("", "_new"))

# List of columns we may update
update_cols = ["posts_scored", "high_quality_rate"] + [f"{c}_mean" for c in int_cols if f"{c}_mean" in out.columns or f"{c}_mean_new" in out.columns]

for col in update_cols:
    new_col = f"{col}_new"
    if new_col not in out.columns:
        continue

    if OVERWRITE_EXISTING:
        # overwrite whenever new value is present
        out[col] = np.where(out[new_col].notna(), out[new_col], out.get(col))
    else:
        # fill only where existing is missing
        if col in out.columns:
            out[col] = out[col].fillna(out[new_col])
        else:
            # if original didn't have the col, just create it from _new
            out[col] = out[new_col]

    out.drop(columns=[new_col], inplace=True)

# Clean up helper key
out.drop(columns=["_handle_norm"], inplace=True)

# ---- Save ----
out.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved → {CSV_OUT}")

# Show a quick preview of key columns if present
preview_cols = [c for c in [HANDLE_COL, "posts_scored", "overall_quality_mean", "high_quality_rate"] if c in out.columns]
print(out[preview_cols].head(10))
