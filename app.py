import streamlit as st
import pandas as pd
from data_preparation import load_and_prepare_data
from recommender import generate_embeddings, create_similarity_matrix, get_movie_recommendations
from model_training import train_model

@st.cache_data
def load_data():
    rm, _, _, movies = load_and_prepare_data()
    return rm, movies

@st.cache_resource
def load_model_and_embeddings():
    try:
        # Try loading saved model
        model = tf.keras.models.load_model('models/recommender_model')
        _, rm, _, _ = load_and_prepare_data()
    except:
        # Train if no saved model exists
        model, rm = train_model()
    
    embeddings = generate_embeddings(model, rm.values)
    item_sim_matrix = create_similarity_matrix(embeddings, rm)
    return item_sim_matrix, rm

def main():
    st.title("Movie Recommender System")

    item_sim_matrix, rm = load_model_and_embeddings()
    _, movies = load_data()

    # Create a list of all movie titles
    all_movies = movies['title'].tolist()
    
    # Create a dropdown (select box) for movie titles
    selected_movie = st.selectbox("Select a movie:", all_movies)

    if selected_movie:
        # Find the selected movie in the dataframe
        selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
        movie_id = selected_movie_data['movieId']

        st.subheader("Selected Movie:")
        st.write(f"{selected_movie_data['title']} (ID: {movie_id})")

        recommendations = get_movie_recommendations(movie_id, item_sim_matrix, movies)

        st.subheader("Recommended Movies:")
        for _, movie in recommendations.iterrows():
            st.write(movie['title'])

if __name__ == "__main__":
    main()