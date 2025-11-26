import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (make sure movies.csv is in the same folder)
movies = pd.read_csv("movies.csv")  # columns: movieId, title, genres

# Combine title + genres into one text field
movies["text"] = movies.apply(
    lambda r: f'{r["title"]} {r["genres"]}' if pd.notnull(r["genres"]) else r["title"],
    axis=1
)

# Vectorize text
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(movies["text"])

# Choose a movie to find similar ones
query_title = "Toy Story (1995)"   # you can change this
idx = movies[movies["title"] == query_title].index

if len(idx) == 0:
    print("Movie not found. Try another title.")
else:
    i = idx[0]
    sims = cosine_similarity(X[i], X).flatten()
    # Sort by similarity, skip the movie itself
    top_idx = sims.argsort()[::-1][1:11]
    print("Movies similar to:", query_title)
    print(movies.loc[top_idx, ["title", "genres"]])