import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

def load_and_prepare_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    
    rm = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    
    train, val = train_test_split(rm, test_size=0.2, random_state=42)
    x_train = train.values
    x_val = val.values
    
    return rm, x_train, x_val, movies

def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return autoencoder

def train_model():
    rm, x_train, x_val, _ = load_and_prepare_data()
    
    input_dim = x_train.shape[1]
    autoencoder = create_autoencoder(input_dim)
    
    history = autoencoder.fit(
        x_train, x_train,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_data=(x_val, x_val)
    )
    
    model_path = 'models/recommender_model'
    autoencoder.save(model_path)
    return autoencoder, rm

if __name__ == "__main__":
    model, rm = train_model()
    print("Model training completed.")