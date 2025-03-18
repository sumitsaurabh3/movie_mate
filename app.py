import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and model
movies_df = pd.read_csv('movies.csv')

# Check if model exists
try:
    with open('recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Ensure 'recommendation_model.pkl' is in the directory.")
    st.stop()

# Calculate similarity matrix if not stored
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


# Streamlit UI
st.title("🎬 Movie Recommendation System MovieMate")

selected_movie = st.selectbox("Choose a movie:", movies_df['title'].values)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader(f"✨ Movies similar to: **{selected_movie}**")
    for movie in recommendations:
        st.write(f"✅ {movie}")
