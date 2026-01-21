import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# === Helper Functions ===
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

# === Load and Preprocess Data ===
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Process fields
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3).apply(collapse)
movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags column
movies['tags'] = movies['overview'] + \
                 movies['genres'].apply(collapse) + \
                 movies['keywords'].apply(collapse) + \
                 movies['cast'] + \
                 movies['crew']

# New dataframe with just title and tags
new = movies[['movie_id', 'title']].copy()
new.loc[:, 'tags'] = movies['tags'].apply(lambda x: " ".join(x)).apply(stem)

# === Vectorization ===
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vectors)

# === Recommendation Logic ===
def recommend(movie):
    try:
        index = new[new['title'] == movie].index[0]
    except IndexError:
        return []
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [new.iloc[i[0]].title for i in sorted_movies]

# === Streamlit UI ===
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title('🎬 Movie Recommendation System')

selected_movie = st.selectbox(
    "Pick a movie to get recommendations:",
    new['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader("You might also like:")
        for title in recommendations:
            st.write("✅", title)
    else:
        st.warning("No recommendations found for this title.")
