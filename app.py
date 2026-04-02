import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import re

# Set page layout
st.set_page_config(page_title="My Show - AI Movie Recommender", layout="wide", page_icon="🍿")

# Optional: Add custom CSS to improve aesthetics
st.markdown("""
<style>
    /* Premium dark mode sleek design overrides */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FF4B4B;
    }
    .poster-container img {
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.3);
        transition: transform 0.3s ease;
    }
    .poster-container img:hover {
        transform: scale(1.05);
    }
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fetch API key (using public reference key to avoid secrets missing warning)
# Please replace with your own in production
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('poster_path'):
                return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except Exception as e:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"

# Preprocessing helpers
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def convert3(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except:
        return []

def fetch_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
    except:
        pass
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

@st.cache_data(show_spinner="Loading AI Core & Datasets...")
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    
    # We keep more raw fields for chatbot filtering
    movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'original_language', 'popularity']]
    movies.dropna(subset=['overview'], inplace=True)
    movies.rename(columns={'id': 'movie_id'}, inplace=True)

    # Convert generic objects
    movies['genres_list'] = movies['genres'].apply(convert) # keep as list for chatbot filters
    movies['genres_tags'] = movies['genres_list']
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3).apply(collapse)
    movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
    movies['overview_tags'] = movies['overview'].apply(lambda x: str(x).split())
    
    movies['tags'] = movies['overview_tags'] + movies['genres_tags'].apply(collapse) + movies['keywords'].apply(collapse) + movies['cast'] + movies['crew']
    
    # Final dataframe for ML
    new = movies[['movie_id', 'title', 'tags', 'vote_average', 'original_language', 'popularity', 'genres_list']].copy()
    new['tags'] = new['tags'].apply(lambda x: " ".join(x)).apply(stem)
    
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return new, similarity

# Load Data
new_df, similarity_matrix = load_data()

# Classic Recommendation Logic (Cosine Similarity)
def recommend_similar(movie):
    try:
        index = new_df[new_df['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return None
    distances = list(enumerate(similarity_matrix[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in sorted_movies:
        movie_row = new_df.iloc[i[0]]
        recommendations.append({
            'title': movie_row.title,
            'id': movie_row.movie_id,
            'rating': movie_row.vote_average
        })
    return recommendations

# Rule-based NLP Chatbot Logic
def chatbot_recommend(user_input):
    in_text = user_input.lower()
    df_filtered = new_df.copy()
    
    # 1. Similarity fallback
    # If they say "similar to Inception" or "like avatar"
    match = re.search(r'(?:similar to|like|movies like) ([a-zA-Z0-9 ]+)', in_text)
    if match:
        movie_query = match.group(1).strip()
        recs = recommend_similar(movie_query)
        if recs:
            return recs, f"Found some great movies similar to '{movie_query}'!"

    # 2. Extract Languages
    if 'bollywood' in in_text or 'hindi' in in_text or 'indian' in in_text:
        df_filtered = df_filtered[df_filtered['original_language'] == 'hi']
    elif 'hollywood' in in_text or 'english' in in_text:
        df_filtered = df_filtered[df_filtered['original_language'] == 'en']

    # 3. Extract Rating
    if any(word in in_text for word in ['top rated', 'high rating', 'best', 'good rating', 'masterpiece']):
        df_filtered = df_filtered[df_filtered['vote_average'] >= 7.5]
    elif 'rating' in in_text:
        df_filtered = df_filtered[df_filtered['vote_average'] >= 6.5]

    # 4. Extract Moods and Genres
    mood_genre_map = {
        'happy': ['Comedy', 'Family', 'Animation'],
        'laugh': ['Comedy'],
        'sad': ['Drama', 'Romance'],
        'cry': ['Drama'],
        'romantic': ['Romance'],
        'scary': ['Horror', 'Thriller'],
        'spooky': ['Horror'],
        'fear': ['Horror'],
        'action': ['Action'],
        'thrilling': ['Action', 'Thriller'],
        'exciting': ['Action', 'Adventure'],
        'sci-fi': ['Science Fiction'],
        'brainy': ['Science Fiction', 'Mystery', 'Documentary'],
        'mystery': ['Mystery'],
        'crime': ['Crime']
    }
    
    matched_genres = []
    for keyword, genres in mood_genre_map.items():
        if keyword in in_text:
            matched_genres.extend(genres)
            
    if matched_genres:
        # Check if movie contains ANY of the matched genres
        def has_genre(movie_genres):
            return any(g in matched_genres for g in movie_genres)
        df_filtered = df_filtered[df_filtered['genres_list'].apply(has_genre)]

    # Final sorting by popularity & rating
    if len(df_filtered) > 0:
        df_filtered = df_filtered.sort_values(by=['popularity', 'vote_average'], ascending=False).head(5)
        recs = []
        for _, row in df_filtered.iterrows():
            recs.append({
                'title': row['title'],
                'id': row['movie_id'],
                'rating': row['vote_average']
            })
        if len(matched_genres) > 0:
            return recs, f"Based on your mood/genre ({', '.join(set(matched_genres))}), here are the top picks!"
        return recs, "Here are some top recommendations matching your criteria!"
    else:
        # Fallback if too strict filter
        top = new_df.sort_values(by='popularity', ascending=False).head(5)
        recs = [{'title': r['title'], 'id': r['movie_id'], 'rating': r['vote_average']} for _, r in top.iterrows()]
        return recs, "I couldn't find an exact match for all your criteria, but here are some very popular choices right now!"

# === UI Architecture ===
st.title("🍿 My Show - AI Movie Recommender")

# Tabs for dual functionality
tab1, tab2 = st.tabs(["Classic Search", "Smart AI Chatbot ✨"])

# TAB 1: Classic Dropdown Search
with tab1:
    st.markdown("### Search a movie to find similar ones:")
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        new_df['title'].values,
        index=0
    )

    if st.button('Show Recommendations', type="primary"):
        recommendations = recommend_similar(selected_movie)
        if recommendations:
            st.success(f"Top 5 movies similar to **{selected_movie}**")
            cols = st.columns(5)
            for col, rec in zip(cols, recommendations):
                with col:
                    st.markdown("<div class='poster-container'>", unsafe_allow_html=True)
                    st.image(fetch_poster(rec['id']), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f"**{rec['title']}**")
                    st.caption(f"⭐ {rec['rating']}/10")
        else:
            st.warning("Sorry, could not find similarities for this movie.")

# TAB 2: AI Rule-based Chatbot
with tab2:
    st.markdown("### Tell me what you're in the mood for...")
    st.caption("Try: *'I am happy and want a highly rated bollywood comedy'* or *'Movies similar to The Dark Knight'*")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history container
    chat_container = st.container(height=400)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "recommendations" in message:
                    recs = message["recommendations"]
                    cols = st.columns(len(recs))
                    for col, rec in zip(cols, recs):
                        with col:
                            st.image(fetch_poster(rec['id']), use_column_width=True)
                            st.markdown(f"**{rec['title']}**")
                            st.caption(f"⭐ {rec['rating']}")

    # Input Box
    prompt = st.chat_input("E.g., Scary hollywood movie with top rating")
    
    if prompt:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate response
        recs, reply_text = chatbot_recommend(prompt)
        
        # Add bot message to state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text,
            "recommendations": recs
        })
        
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply_text)
                if recs:
                    cols = st.columns(len(recs))
                    for col, rec in zip(cols, recs):
                        with col:
                            st.image(fetch_poster(rec['id']), use_column_width=True)
                            st.markdown(f"**{rec['title']}**")
                            st.caption(f"⭐ {rec['rating']}")
