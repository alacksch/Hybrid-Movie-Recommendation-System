import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("../data/movies.csv")
    ratings = pd.read_csv("../data/ratings.csv")

    movies["genres"] = movies["genres"].fillna("")
    movies["genres_str"] = movies["genres"].str.replace("|", " ")

    return movies, ratings

@st.cache_resource
def load_model():
    with open("svd_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_predictions():
    with open("predictions.pkl", "rb") as f:
        return pickle.load(f)

movies, ratings = load_data()
model = load_model()
predictions = load_predictions()

# =========================
# CONTENT ENGINE
# =========================
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres_str"])
similarity = cosine_similarity(tfidf_matrix)

# =========================
# HYBRID FUNCTION
# =========================
scaler = MinMaxScaler()

def recommend(user_id, movie_title, top_n=10, alpha=0.6):
    idx = movies[movies["title"] == movie_title].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:51]

    indices = [i[0] for i in scores]
    candidates = movies.iloc[indices][["movieId", "title"]].copy()

    collab_scores = []
    for _, row in candidates.iterrows():
        pred = model.predict(user_id, row["movieId"]).est
        collab_scores.append(pred)

    candidates["collab_score"] = collab_scores
    candidates["collab_score"] = scaler.fit_transform(candidates[["collab_score"]])

    candidates["content_score"] = np.linspace(1, 0, len(candidates))

    candidates["final_score"] = (
        alpha * candidates["collab_score"] +
        (1 - alpha) * candidates["content_score"]
    )

    return candidates.sort_values("final_score", ascending=False).head(top_n)

# =========================
# METRICS
# =========================
def compute_metrics(predictions, threshold=3.5):
    y_true, y_pred = [], []

    for p in predictions:
        y_true.append(1 if p.r_ui >= threshold else 0)
        y_pred.append(1 if p.est >= threshold else 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    cm = confusion_matrix(y_true, y_pred)

    return precision, recall, f1, cm

precision, recall, f1, cm = compute_metrics(predictions)

# =========================
# UI
# =========================
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")

page = st.sidebar.selectbox("Navigation", ["Recommend", "Metrics"])

# =========================
# RECOMMEND PAGE
# =========================
if page == "Recommend":

    st.header("Get Movie Recommendations")

    user_id = st.number_input("User ID", min_value=1, value=1)

    movie_title = st.selectbox(
        "Select a Movie",
        movies["title"].sort_values().unique()
    )

    alpha = st.slider("Hybrid Weight (Collaborative Importance)", 0.0, 1.0, 0.6)

    if st.button("Recommend"):

        results = recommend(user_id, movie_title, alpha=alpha)

        st.subheader("Top Recommendations")
        st.dataframe(results[["title", "final_score"]])

# =========================
# METRICS PAGE
# =========================
elif page == "Metrics":

    st.header("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)