# Streamlit Sentiment App

This app lets visitors type an Amazon-style food review and see a model prediction.

## Quick start

1. Clone or download this repository.
2. Download the Kaggle dataset from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews and put `database.sqlite` in the repository root.
3. In a terminal in the repository root, run:

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

The training script creates `models/sentiment_pipeline.joblib` locally. Neither the SQLite database nor the model artifact is committed to GitHub.

**Security:** Only load model files you created or trust. Joblib/pickle can execute code when loading untrusted artifacts.

## Notes

The app model is trained by `train_model.py`; it is not the already-fitted model from the Jupyter session. Its accuracy may differ from the notebook's 90.2% because sampling and preprocessing differ. Run training to see this model's own measured accuracy and macro F1.

To deploy publicly, arrange a trusted model artifact in the deployment environment or add a safe training/build process. The app will show setup instructions until a model exists.
