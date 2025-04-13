import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Set page config
st.set_page_config(page_title="MovieMate 🎬", page_icon="🍿", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            text-align: center;
            color: #FF4B4B;
        }
        .movie-card {
            background-color: #1E1E1E;
            padding: 1em;
            margin: 0.5em 0;
            border-radius: 10px;
            color: white;
            box-shadow: 0 0 5px rgba(255,255,255,0.1);
        }
        .movie-card:hover {
            background-color: #333333;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🍿 MovieMate")

st.sidebar.markdown("---")

# Load dataset
movies_df = pd.read_csv('movies.csv')

# Ensure 'tags' column exists
if 'tags' not in movies_df.columns:
    st.error("❌ 'tags' column missing from movies.csv!")
    st.stop()

# Check model existence or create a new one
try:
    with open('recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("✅ Model loaded.")
except FileNotFoundError:
    st.sidebar.warning("🔄 Model not found. Generating...")
    vectorizer = TfidfVectorizer()
    model = vectorizer.fit_transform(movies_df['tags'])
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.sidebar.success("✅ Model generated and saved.")

# Ensure model is numeric
if not isinstance(model, np.ndarray):
    model = model.toarray()

# Calculate similarity matrix
similarity = cosine_similarity(model)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies_df['title'].values:
        return ["Movie not found!"]
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    similar_movies = list(enumerate(similarity[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    recommended_movies = [movies_df.iloc[i[0]].title for i in similar_movies[1:6]]
    return recommended_movies

# App Title
st.markdown('<div class="title">🎬 MovieMate Recommender</div>', unsafe_allow_html=True)
st.markdown("#### Discover movies similar to your favorites!")

# Movie Selector
selected_movie = st.selectbox("🎥 Choose a movie:", movies_df['title'].values)

if st.button("🔍 Show Recommendations"):
    recommendations = recommend(selected_movie)

    if recommendations[0] == "Movie not found!":
        st.error("🚫 Movie not found in the dataset.")
    else:
        st.markdown(f"### ✨ Because you liked **{selected_movie}**, you might also enjoy:")
        for movie in recommendations:
            st.markdown(f"""
                <div class="movie-card">
                    ✅ {movie}
                </div>
            """, unsafe_allow_html=True)
