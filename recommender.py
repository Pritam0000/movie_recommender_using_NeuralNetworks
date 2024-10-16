from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def generate_embeddings(model, data):
    intermediate_model = Model(model.input, model.layers[3].output)
    return intermediate_model.predict(data)

def create_similarity_matrix(embeddings, rm):
    similarity_matrix = cosine_similarity(embeddings)
    return pd.DataFrame(similarity_matrix, index=rm.index, columns=rm.index)

def get_movie_recommendations(movie_id, item_sim_matrix, movies, n=10):
    similar_movies = item_sim_matrix[movie_id].sort_values(ascending=False).head(n+1).index[1:]
    return movies[movies.movieId.isin(similar_movies)]