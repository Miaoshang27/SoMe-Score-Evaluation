
# SoMe Score — Deployable Pipeline

## Files
- `engineer_some.py` — leakage-safe feature engineering transformer.
- `train_and_save.py` — trains, calibrates, tunes thresholds, evaluates, and exports `artifacts/some_model.joblib` + `metrics.json` + `MODEL_CARD.md`.
- `serve.py` — FastAPI inference service that loads the saved bundle.
- `requirements.txt` — pinned versions.

## Quick start (local)
```bash
pip install -r requirements.txt

# Train (expects CSV with columns: engagement_rate, follower_count_c, posting_frequency_c, if_success)
python train_and_save.py --csv /path/to/your_data.csv --out artifacts

# Serve
uvicorn serve:app --host 0.0.0.0 --port 8080

# Test
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"engagement_rate":0.02,"follower_count_c":1200,"posting_frequency_c":1.5}]}' | jq .
```

## Notes
- Thresholds are tuned on the validation split to maximize F1 for `t_review` and to reach `min_launch_precision` for `t_launch`.
- Calibration uses isotonic on a held-out calibration split from the training set.
- The saved bundle includes the full pipeline + thresholds + basic metrics and version.
