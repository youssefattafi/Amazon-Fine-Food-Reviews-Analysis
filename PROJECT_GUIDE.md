# Amazon Fine Food Reviews — Project Guide

## Project goal
Analyze patterns in Amazon Fine Food Reviews and train a baseline text classifier for rating-derived sentiment.

## Data source
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Download `database.sqlite` separately and put it in the repository root. The dataset is not committed to GitHub.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/amazon_reviews_ml_analysis.ipynb
```

Windows activation: `.venv\Scripts\activate`. You can instead set `AMAZON_REVIEWS_DB` to the path of the downloaded database. Run Jupyter from the repository root.

## Contents
- Original uploaded notebook at repository root: `amazon_fine_food_reviews_analysis.ipynb` (preserved)
- Improved notebook: `notebooks/amazon_reviews_ml_analysis.ipynb`
- Reproducible EDA: score distribution, review lengths, frequently reviewed products
- Sentiment baseline: TF-IDF + Logistic Regression with stratified held-out test set
- Evaluation: majority baseline, accuracy, macro F1, per-class precision/recall, confusion matrix

## Responsible interpretation
Reviews are not purchases. Sentiment labels are derived from star scores, not independently annotated. Three-star reviews are excluded. Performance metrics are generated when the notebook runs and are not fabricated in the repository.

## Status
Code prepared; execution on the complete Kaggle database has not been verified. Review and record metrics after running locally.
