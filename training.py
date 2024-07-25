import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
data_path = 'playlist_dataset.csv'  # Update with your dataset path
data = pd.read_csv(data_path)

# Encode categorical features
label_encoders = {}
for col in ['artist_genres']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
features = ['danceability', 'energy', 'loudness', 'mode', 'speechiness',
             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_genres']
X = data[features]
y = data['track_name']  # Assuming 'track_name' is your target

# Optionally, use a subset of the data if it's too large
if len(data) > 1000:  # Adjust the number as needed
    data = data.sample(1000, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoders
recommendation_model = {
    'model': model,
    'label_encoders': label_encoders,
    'music_data': data  # Optionally save a subset of the data
}

with open('recommendation.pkl', 'wb') as f:
    pickle.dump(recommendation_model, f)

# Optional: Compress the pickle file to reduce size
import gzip
with gzip.open('recommendation.pkl.gz', 'wb') as f:
    pickle.dump(recommendation_model, f)
