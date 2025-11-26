# file: app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

@st.cache_resource
def build_tfidf(movies):
    movies["text"] = movies.apply(
        lambda r: f'{r["title"]} {r["genres"]}' if pd.notnull(r["genres"]) else r["title"],
        axis=1
    )
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(movies["text"])
    return tfidf, X

movies, ratings = load_data()

st.title("🎬 Movie Recommender System")
st.write("Choose a method: Popularity, Genre-based, or Similar-to-a-Movie.")

tab1, tab2, tab3 = st.tabs(["Popularity", "Genre", "Similar Movie"])

# ---------------- POPULARITY TAB ----------------
with tab1:
    st.subheader("Top movies by rating (min 50 ratings)")
    agg = ratings.groupby("movieId").agg(
        count=("rating", "count"),
        mean=("rating", "mean")
    ).reset_index()
    agg = agg.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")

    min_count = st.slider("Minimum number of ratings", 10, 200, 50, step=10, key="popularity_min_count")
    top_n_popularity = st.slider("Number of recommendations", 5, 50, 10, step=5, key="popularity_top_n")

    popular = agg[agg["count"] >= min_count].sort_values(["mean", "count"], ascending=False)
    st.dataframe(popular.head(top_n_popularity)[["title", "genres", "mean", "count"]])

# ---------------- GENRE TAB ----------------
with tab2:
    st.subheader("Recommend by genre")
    movies["genres_list"] = movies["genres"].apply(lambda g: g.split("|") if isinstance(g, str) else [])
    all_genres = sorted({g for gl in movies["genres_list"] for g in gl if g != "(no genres listed)"})

    genre = st.selectbox("Choose a genre", all_genres, key="genre_select")
    top_n_genre = st.slider("Number of recommendations", 5, 50, 10, step=5, key="genre_top_n")

    rec = movies[movies["genres_list"].apply(lambda gl: genre in gl)]
    st.dataframe(rec.head(top_n_genre)[["title", "genres"]])

# ---------------- SIMILAR MOVIE TAB ----------------
with tab3:
    st.subheader("Recommend similar to a specific movie")
    tfidf, X = build_tfidf(movies)

    title = st.selectbox("Choose a movie", movies["title"].unique(), key="similar_select")
    top_n_similar = st.slider("Number of recommendations", 5, 50, 10, step=5, key="similar_top_n")

    idx = movies[movies["title"] == title].index[0]
    sims = cosine_similarity(X[idx], X).flatten()
    top_idx = sims.argsort()[::-1][1:top_n_similar+1]

    st.dataframe(movies.loc[top_idx, ["title", "genres"]])