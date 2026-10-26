"""Interactive sentiment prediction demo. Run with: streamlit run streamlit_app.py"""
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "sentiment_pipeline.joblib.gz"

st.set_page_config(page_title="Amazon Review Sentiment AI", page_icon="🛍️", layout="centered")
st.title("🛍️ Amazon Review Sentiment AI")
st.caption("TF-IDF + Logistic Regression | Amazon Fine Food Reviews")
st.write("Enter a product review to estimate whether its language resembles reviews rated 1–2 stars (negative) or 4–5 stars (positive).")

if not MODEL.is_file():
    st.warning("Trained model not deployed yet. Run `python export_model.py` locally, then upload `models/sentiment_pipeline.joblib.gz` to this repository.")
    st.code("python train_model.py\npython export_model.py\nstreamlit run streamlit_app.py", language="bash")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

model = load_model()
examples = {
    "Choose an example": "",
    "Positive": "Delicious coffee, fresh flavor, and fast delivery. I would buy it again.",
    "Negative": "The food arrived stale and tasted terrible. Very disappointing.",
}
choice = st.selectbox("Example reviews", list(examples))
review = st.text_area("Your review", value=examples[choice], height=150,
                      placeholder="Write a review here...")
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.error("Please enter some review text.")
    else:
        prediction = int(model.predict([review])[0])
        probabilities = model.predict_proba([review])[0]
        label = "Positive" if prediction == 1 else "Negative"
        confidence = float(max(probabilities))
        st.subheader(f"{'😊' if prediction else '😟'} Predicted sentiment: {label}")
        st.metric("Model confidence estimate", f"{confidence:.1%}")
        st.progress(confidence)
        st.caption("This is a model probability estimate, not a guarantee of correctness or a verified customer opinion.")

with st.expander("About this model"):
    st.markdown("""
    - Dataset: [SNAP Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
    - Labels: 1–2 stars = negative; 4–5 stars = positive; 3-star reviews excluded.
    - Algorithm: TF-IDF (unigrams and bigrams) + balanced Logistic Regression.
    - The app uses an exported trained model. Reported notebook metrics are **not** automatically the metrics of this separately trained artifact.
    - The dataset records reviews, not confirmed purchases.
    """)
