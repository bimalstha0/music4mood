import streamlit as st
import helper
import cv2
import numpy as np

st.title('Face Recognition Demo')


st.text('Music4Mood is a recommender system which uses computer vision and CNN to detect face informations and use those emoions to recommend music to the user.')
st.selectbox('Which model would you like to use?',
             ('Custom Model v1.0','Emotion-FerPlus'))
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)

uploaded_file = st.file_uploader("Upload an image",type=['png','jpg','jpeg','webp'])

if uploaded_file:
    st.image(uploaded_file)
    faces = helper.get_face_from_upload(uploaded_file)
    if not faces:
        st.write('Oops! No faces detected.')
    else:
        st.write(len(faces),'faces detected.')

    predit_btn = st.button('Predict mood')

    if predit_btn:
        for face in faces:
            st.image(face)
            st.write(helper.predict(face))