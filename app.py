import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast

# Preprocessing helpers
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    for i in ast.literal_eval(obj)[:3]:
        L.append(i['name'])
    return L

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Load and preprocess data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3).apply(collapse)
movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'].apply(collapse) + movies['keywords'].apply(collapse) + movies['cast'] + movies['crew']
new = movies[['movie_id', 'title']]
new['tags'] = movies['tags'].apply(lambda x: " ".join(x)).apply(stem)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation logic
def recommend(movie):
    try:
        index = new[new['title'] == movie].index[0]
    except IndexError:
        return []
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [new.iloc[i[0]].title for i in sorted_movies]

# Streamlit UI
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
        st.warning("Movie not found or no similar movies available.")
