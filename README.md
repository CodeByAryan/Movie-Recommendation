# 🎬 Movie Recommender System

An interactive **Movie Recommendation Web App** built with **Streamlit** and powered by the **MovieLens dataset**.  
This project demonstrates multiple recommendation techniques — from simple popularity rankings to genre filtering and content‑based similarity — all wrapped in a clean, professional UI.

---

## ✨ Features
- 📈 **Popularity‑based recommendations**: Shows top‑rated movies with configurable minimum rating counts.  
- 🎭 **Genre‑based filtering**: Pick a genre and instantly see curated recommendations.  
- 🔍 **Similar movie finder**: Select a movie and discover others with similar content using **TF‑IDF + cosine similarity**.  
- 🖥️ **Interactive UI**: Built with Streamlit tabs, sliders, and dropdowns for a smooth user experience.  
- 🚀 **Scalable design**: Easily extendable to include posters, ratings visualization, or deployment online.  

---

## 🛠️ Tech Stack
- **Python 3.13+**  
- **Streamlit** (for web UI)  
- **Pandas** (data handling)  
- **Scikit‑learn** (TF‑IDF vectorization & similarity)  
- **MovieLens dataset** (`movies.csv`, `ratings.csv`, `links.csv`)  

---

## 📊 Dataset
This project uses the **MovieLens dataset** (commonly available at [GroupLens](https://grouplens.org/datasets/movielens/)).  
Files required:
- `movies.csv`  
- `ratings.csv`  
- `links.csv` (optional, for TMDb poster integration)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
