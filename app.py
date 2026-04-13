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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }

    /* Premium dark mode sleek design overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #0E1117);
        color: #FAFAFA;
    }

    h1, h2, h3 {
        color: #FF4B4B;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #FF4B4B, #FF7676);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        background: linear-gradient(90deg, #FF7676, #FF4B4B);
    }

    /* Card Styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }

    .movie-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #FF4B4B;
        transform: translateY(-5px);
    }

    .poster-container img {
        border-radius: 10px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px;
    }

    /* Watch Badge */
    .watch-badge {
        display: inline-block;
        padding: 4px 8px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(14, 17, 23, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Fetch API key (using public reference key to avoid secrets missing warning)
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

def fetch_watch_providers(movie_id, region='IN'):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            region_data = results.get(region, {})
            
            providers = []
            # Combine flatrate, rent, and buy
            for key in ['flatrate', 'rent', 'buy']:
                if key in region_data:
                    for p in region_data[key]:
                        providers.append({
                            'name': p['provider_name'],
                            'logo': "https://image.tmdb.org/t/p/original" + p['logo_path'],
                            'type': key
                        })
            
            # De-duplicate by name
            unique_providers = {p['name']: p for p in providers}.values()
            
            # The JustWatch link
            link = region_data.get('link', None)
            return list(unique_providers), link
    except:
        pass
    return [], None

def fetch_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for v in data.get('results', []):
                if v['type'] == 'Trailer' and v['site'] == 'YouTube':
                    return f"https://www.youtube.com/watch?v={v['key']}"
    except:
        pass
    return None

def fetch_cast(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            cast = data.get('cast', [])[:5]
            return [{
                'name': c['name'],
                'char': c['character'],
                'photo': "https://image.tmdb.org/t/p/w200" + c['profile_path'] if c['profile_path'] else None
            } for c in cast]
    except:
        pass
    return []

def fetch_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'runtime': data.get('runtime', 0),
                'release': data.get('release_date', 'Unknown'),
                'overview': data.get('overview', ''),
                'tagline': data.get('tagline', '')
            }
    except:
        pass
    return None

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

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_movie" not in st.session_state:
    st.session_state.current_movie = None

def go_to_details(movie_id, movie_title, movie_rating):
    st.session_state.current_movie = {'id': movie_id, 'title': movie_title, 'rating': movie_rating}
    st.session_state.page = "details"
    st.rerun()

def go_to_home():
    st.session_state.page = "home"
    st.session_state.current_movie = None
    st.rerun()

# Sidebar Controls
st.sidebar.title("Configuration ⚙️")
region = st.sidebar.selectbox("Select Region for Watch Providers", ['IN', 'US', 'GB', 'CA', 'AU'], index=0)

if st.sidebar.button("🏠 Home", use_container_width=True):
    go_to_home()

if st.sidebar.button("🎲 Surprise Me!", use_container_width=True):
    st.session_state.surprise_movie = new_df.sample(1).iloc[0]

if "surprise_movie" in st.session_state:
    random_movie = st.session_state.surprise_movie
    st.sidebar.info(f"How about watching: **{random_movie['title']}**?")
    if st.sidebar.button("See Details", key="surprise_details", use_container_width=True):
        go_to_details(random_movie['movie_id'], random_movie['title'], random_movie['vote_average'])

def display_movie_card(rec, key_prefix=""):
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <div class="poster-container" style="text-align: center;">
                <img src="{fetch_poster(rec['id'])}" style="width: 100%; border-radius: 10px;">
            </div>
            <div style="text-align: center; margin-top: 10px; min-height: 80px;">
                <h4 style="margin: 0; font-size: 1rem; color: #FAFAFA;">{rec['title']}</h4>
                <p style="color: #FF4B4B; margin: 5px 0; font-weight: bold;">⭐ {rec['rating']}/10</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ View Details", key=f"btn_{key_prefix}_{rec['id']}", use_container_width=True):
            go_to_details(rec['id'], rec['title'], rec['rating'])

def render_detail_page():
    movie = st.session_state.current_movie
    if not movie:
        go_to_home()
        return

    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.image(fetch_poster(movie['id']), use_container_width=True)
        if st.button("⬅️ Back to Recommendations", use_container_width=True):
            go_to_home()

    with col2:
        st.title(movie['title'])
        st.markdown(f"### ⭐ {movie['rating']}/10")
        
        details = fetch_details(movie['id'])
        if details:
            if details['tagline']:
                st.markdown(f"*{details['tagline']}*")
            st.write(f"📅 **Released:** {details['release']} | ⏳ **Runtime:** {details['runtime']} min")
            st.markdown("#### Overview")
            st.write(details['overview'])
        
        # Watch Providers
        providers, jw_link = fetch_watch_providers(movie['id'], region=region)
        if providers:
            st.markdown("---")
            st.markdown("#### 📺 Available On")
            p_cols = st.columns(min(len(providers), 6))
            for i, p in enumerate(providers):
                if i < 6:
                    with p_cols[i]:
                        st.image(p['logo'], use_container_width=True)
                        st.caption(p['name'])
            if jw_link:
                st.link_button("Check all options on JustWatch ↗️", jw_link)
        else:
            st.info(f"No streaming info available for {region}. Try changing the region in the sidebar.")

        # Trailer
        trailer_url = fetch_trailer(movie['id'])
        if trailer_url:
            st.markdown("---")
            st.markdown("#### 🎬 Official Trailer")
            st.video(trailer_url)

        # Cast
        cast = fetch_cast(movie['id'])
        if cast:
            st.markdown("---")
            st.markdown("#### 🎭 Top Cast")
            cast_cols = st.columns(5)
            for i, c in enumerate(cast):
                with cast_cols[i]:
                    if c['photo']:
                        st.image(c['photo'], use_container_width=True)
                    st.caption(f"**{c['name']}**")
                    st.caption(f"*{c['char']}*")

def render_main_page():
    st.title("🍿 My Show - AI Movie Recommender")
    
    # Tabs for dual functionality
    tab1, tab2, tab3 = st.tabs(["Classic Search", "Smart AI Chatbot ✨", "Trending 🔥"])

    # TAB 1: Classic Dropdown Search
    with tab1:
        st.markdown("### Search a movie to find similar ones:")
        selected_movie = st.selectbox(
            "Type or select a movie from the dropdown",
            new_df['title'].values,
            index=0,
            key="search_box"
        )

    if st.button('Show Recommendations', type="primary", key="search_btn"):
        st.session_state.search_recs = recommend_similar(selected_movie)
    
    if "search_recs" in st.session_state:
        recommendations = st.session_state.search_recs
        if recommendations:
            st.success(f"Top picks based on your search:")
            cols = st.columns(5)
            for i, (col, rec) in enumerate(zip(cols, recommendations)):
                with col:
                    display_movie_card(rec, key_prefix=f"classic_{i}")
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
        chat_container = st.container(height=500)
        
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "recommendations" in message:
                        recs = message["recommendations"]
                        cols = st.columns(len(recs))
                        for j, (col, rec) in enumerate(zip(cols, recs)):
                            with col:
                                display_movie_card(rec, key_prefix=f"chat_{i}_{j}")

        # Input Box
        prompt = st.chat_input("E.g., Scary hollywood movie with top rating")
        
        if prompt:
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            recs, reply_text = chatbot_recommend(prompt)
            
            # Add bot message to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": reply_text,
                "recommendations": recs
            })
            st.rerun()

    # TAB 3: Trending Movies
    with tab3:
        st.markdown("### Global Trending Movies")
        try:
            url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
            res = requests.get(url).json()
            trending = res.get('results', [])[:10]
            
            cols1 = st.columns(5)
            for i, t in enumerate(trending[:5]):
                with cols1[i]:
                    display_movie_card({'id': t['id'], 'title': t['title'], 'rating': t['vote_average']}, key_prefix=f"trending_{i}")
            
            cols2 = st.columns(5)
            for i, t in enumerate(trending[5:]):
                with cols2[i]:
                    display_movie_card({'id': t['id'], 'title': t['title'], 'rating': t['vote_average']}, key_prefix=f"trending_v2_{i}")
        except:
            st.error("Failed to load trending movies.")

# Routing Logic
if st.session_state.page == "details":
    render_detail_page()
else:
    render_main_page()
