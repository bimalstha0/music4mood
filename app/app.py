import streamlit as st
import helper
import pandas as pd
import random
import ast

# Load and process music data to get top artists by total popularity
music_data = pd.read_csv('final.csv')  # Assumes CSV has 'artists' and 'popularity' columns

# Convert 'artists' column from string representation of list to actual list, if needed
music_data['artists'] = music_data['artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Calculate total popularity for each artist by exploding the artists column
all_artists = music_data.explode('artists')
artist_popularity = all_artists.groupby('artists')['popularity'].sum().sort_values(ascending=False)
top_15_artists = artist_popularity.head(15).index.tolist()

st.title('Face Recognition Music Recommender')

st.write('Music4Mood is a recommender system that uses computer vision and CNN to detect facial emotions and recommend music based on mood.')

# Image input
img_file_buffer = st.camera_input("Take a picture")
image = None
if img_file_buffer is not None:
    image = img_file_buffer
st.write('Hey there! Let’s get some music recommendations for you.')
if not image:
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_file:
        image = uploaded_file
        st.image(uploaded_file)

# Artist selection
if image:
    # Initialize an empty list to store selected artists in session state
    if "selected_artists" not in st.session_state:
        st.session_state.selected_artists = []

    # Buttons for top 15 artists
    st.write("Choose from top artists:")
    col1, col2, col3 = st.columns(3)
    for idx, artist in enumerate(top_15_artists):
        if idx % 3 == 0:
            if col1.button(artist, key=artist):
                if artist not in st.session_state.selected_artists:
                    st.session_state.selected_artists.append(artist)
        elif idx % 3 == 1:
            if col2.button(artist, key=artist):
                if artist not in st.session_state.selected_artists:
                    st.session_state.selected_artists.append(artist)
        else:
            if col3.button(artist, key=artist):
                if artist not in st.session_state.selected_artists:
                    st.session_state.selected_artists.append(artist)

    # Display selected artists
    st.write("Selected artists:", st.session_state.selected_artists)

    # Text input for additional artist search
    custom_artist = st.text_input("Or search for another artist")
    if custom_artist and custom_artist not in st.session_state.selected_artists:
        st.session_state.selected_artists.append(custom_artist)
    
    # Process the image and predict the mood only when the "Predict Mood and Recommend Song" button is clicked
    if st.button('Predict Mood and Recommend Song'):
        # Process the image to detect faces and predict mood
        faces = helper.get_face_from_upload(image)
        if len(faces) > 1:
            st.write(f"{len(faces)} faces detected.")
        elif len(faces) == 1:
            st.write("1 face detected.")
        else:
            st.write("Oops! No faces detected.")

        # If faces are detected, predict the mood and recommend a song
        if faces:
            # Predict the mood from the image
            mood = helper.predict(faces[0])
            st.write(f"Predicted mood: {mood}")

            # Filter music data to include only songs by selected artists and matching mood
            def artist_match(song_artists, selected_artists):
                return any(artist in song_artists for artist in selected_artists)

            filtered_songs = music_data[music_data['mood'] == mood]
            filtered_songs = filtered_songs[filtered_songs['artists'].apply(lambda song_artists: artist_match(song_artists, st.session_state.selected_artists))].reset_index(drop=True)

            # Recommend a song if any songs match the criteria
            if not filtered_songs.empty:
                recommended_song = filtered_songs.sort_values(by='popularity', ascending=False).iloc[random.randint(0,len(filtered_songs))]
                st.write(f"Recommended song for mood '{mood}': '{recommended_song['name']}' by {recommended_song['artists']}.")
            else:
                st.write(f"No songs found for mood '{mood}' by the selected artists.")

