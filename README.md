# Movie Recommender System

This project implements a movie recommender system using autoencoders and deploys it as a Streamlit app.

## Setup

1. Clone this repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the MovieLens dataset and place 'ratings.csv' and 'movies.csv' in the 'data/' directory.

## Usage

1. Prepare the data and train the model:
   ```
   python src/data_preparation.py
   python src/model_training.py
   ```
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open your web browser and go to the URL displayed in the terminal.

## Project Structure

- `data/`: Contains the dataset files.
- `models/`: Stores the trained autoencoder model.
- `src/`: Contains the source code for data preparation, model training, and recommendation logic.
- `app.py`: The main Streamlit application.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, containing project information and instructions.
