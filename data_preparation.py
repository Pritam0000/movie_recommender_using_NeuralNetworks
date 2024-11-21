import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    
    rm = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    
    train, val = train_test_split(rm, test_size=0.2, random_state=42)
    x_train = train.values
    x_val = val.values
    
    return rm, x_train, x_val, movies
