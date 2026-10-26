# Deploy the Amazon Review Sentiment AI app

The app runs locally and is ready for Streamlit Community Cloud **after you add the trained model artifact**. The GitHub repository currently contains the code but not the model weights.

## Step 1: Export the model on your Mac

In the new project folder containing `database.sqlite`, run:

```bash
python -m pip install -r requirements.txt
python train_model.py
python export_model.py
```

The exporter compresses the trained model to `models/sentiment_pipeline.joblib.gz`. It does not upload the 373 MB SQLite dataset. If you already trained the model, you can skip `python train_model.py`.

## Step 2: Upload the exported model to GitHub

Open the repository's `models` folder (or use GitHub's **Add file → Upload files** at the repository root and upload a folder containing the artifact). Upload **only** `models/sentiment_pipeline.joblib.gz` and commit. GitHub's web upload limit is 25 MiB per file. If the artifact is larger, stop and use a separate model-hosting approach instead.

Never upload `database.sqlite` or the uncompressed `.joblib` model.

## Step 3: Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app → Deploy a public app from GitHub**.
3. Repository: `youssefattafi/Amazon-Fine-Food-Reviews-Analysis`; branch: `main`; main file: `streamlit_app.py`.
4. Click **Deploy** and wait for the app to build.
5. Test both positive and negative sample reviews; share the resulting public URL in your resume or portfolio.

## Notes

The model is trained locally, not on Streamlit's server. The SQLite dataset is not required for inference. The training script's metrics may differ from the executed notebook's 90.2% accuracy. Joblib files must only be loaded from trusted sources, because loading them can execute code. For reproducibility, use compatible scikit-learn versions between training and deployment.
