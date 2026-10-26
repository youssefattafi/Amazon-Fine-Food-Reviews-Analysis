"""Compress a locally trained sentiment model for GitHub/Streamlit deployment."""
from pathlib import Path
import joblib

root = Path(__file__).resolve().parent
source = root / "models" / "sentiment_pipeline.joblib"
target = root / "models" / "sentiment_pipeline.joblib.gz"
if not source.exists():
    raise SystemExit("Model not found. Run python train_model.py first.")
model = joblib.load(source)  # Only load your own trusted model file.
joblib.dump(model, target, compress=("gzip", 3))
size = target.stat().st_size / (1024 * 1024)
print(f"Exported {target} ({size:.2f} MiB)")
if size > 25:
    print("Warning: file exceeds GitHub browser upload limit (25 MiB).")
