import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the recommendation model
with open('recommendation.pkl', 'rb') as f:
    recommendation_model = pickle.load(f)

model = recommendation_model['model']
label_encoders = recommendation_model['label_encoders']
music_data = recommendation_model['music_data']

# Function to make recommendations based on music name
def make_name_recommendations(music_name):
    # Convert input music name to lowercase and remove extra spaces
    music_name = music_name.lower().replace(" ", "")

    # Convert track names to lowercase and remove spaces for case-insensitive and space-insensitive comparison
    music_data['track_name_clean'] = music_data['track_name'].str.lower().str.replace(" ", "")

    if music_name not in music_data['track_name_clean'].values:
        return None, None

    # Get the feature vector of the input music
    input_music = music_data[music_data['track_name_clean'] == music_name]
    input_features = input_music[['danceability', 'energy', 'loudness', 'mode', 'speechiness',
                                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_genres']]

    # Get feature vectors of all songs
    all_features = music_data[['danceability', 'energy', 'loudness', 'mode', 'speechiness',
                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_genres']]

    # Compute similarity between input song and all other songs
    similarities = cosine_similarity(input_features, all_features)
    similar_indices = similarities[0].argsort()[-10:][::-1]  # Get top 10 similar indices

    recommended_tracks = music_data.iloc[similar_indices]['track_name'].values
    recommended_urls = music_data.iloc[similar_indices]['playlist_url'].values

    return recommended_tracks, recommended_urls

# Streamlit app
st.title("Music Recommender")

# User input - music name
music_name = st.text_input("Enter a music name:")
if music_name:
    recommended_tracks, recommended_urls = make_name_recommendations(music_name)
    if recommended_tracks is None:
        st.write("Music name not found in the database. Please try another name.")
    else:
        st.write("Recommended Songs and Playlists:")
        for track, url in zip(recommended_tracks, recommended_urls):
            st.write(f"Song: {track} | Playlist: [Link]({url})")
