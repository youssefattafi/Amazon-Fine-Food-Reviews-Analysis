# Amazon Fine Food Reviews — Exploratory Data Analysis

A Jupyter notebook project exploring Amazon Fine Food Reviews using Python, SQLite, pandas, and visualizations.

## Dataset

Source: [SNAP Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

The original archive includes `database.sqlite` (~373 MB), which is **not included** in this repository to avoid GitHub file-size limits. Download the dataset from Kaggle and place `database.sqlite` at the project root (or adjust the notebook's database path).

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/amazon_fine_food_reviews_analysis.ipynb
```

## Project files

- `notebooks/amazon_fine_food_reviews_analysis.ipynb`: analysis notebook from the supplied archive (upload pending).
- `requirements.txt`: commonly used analysis dependencies.
- `.gitignore`: excludes local datasets, notebook checkpoints, and Python environments.

## Notes

This notebook has not been fully tested or refactored. Some cells reference Windows-specific paths (including a local image) and may need updating to run on a different machine.

## Dataset credit

Dataset: SNAP Amazon Fine Food Reviews, distributed via Kaggle. Consult the dataset page for provenance and applicable terms.
