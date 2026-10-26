# Amazon Fine Food Reviews — EDA & Sentiment Classification

A Python data science portfolio project exploring review ratings and text, plus a TF-IDF + Logistic Regression baseline for predicting rating-derived sentiment.

## Project notebooks

- [Improved EDA + ML notebook](notebooks/amazon_reviews_ml_analysis.ipynb) — reproducible analysis, visualizations, stratified train/test split, accuracy, macro F1, classification report and confusion matrix.
- [Original notebook](amazon_fine_food_reviews_analysis.ipynb) — preserved as originally uploaded.

## Dataset

[SNAP Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Download `database.sqlite` and put it in the repository root; the large database is not committed. Alternatively set the `AMAZON_REVIEWS_DB` environment variable to its path.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/amazon_reviews_ml_analysis.ipynb
```

On Windows activate with `.venv\Scripts\activate`. Launch Jupyter from the repository root.

## Methodology

- Validate and clean reviews; remove missing and duplicate text.
- Explore rating distribution, review lengths and most frequently reviewed products.
- Create positive (4–5 stars) and negative (1–2 stars) proxy labels; omit neutral 3-star ratings.
- Train TF-IDF + balanced Logistic Regression on training data only.
- Evaluate against a majority-class baseline on a held-out test set.

**Interpretation:** These are reviews, not verified purchases. Sentiment labels come from star ratings. The improved notebook has not yet been executed against the full dataset, so no model scores are claimed here.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for details and limitations.
