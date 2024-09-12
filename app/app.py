import streamlit as st
import helper

st.title('Face Recognition Demo')


st.write('Music4Mood is a recommender system which uses computer vision and CNN to detect face informations and use those emoions to recommend music to the user.')
st.selectbox('Which model would you like to use?',
             ('Custom Model v1.0','Emotion-FerPlus'))
img_file_buffer = st.camera_input("Take a picture")
image = None
if img_file_buffer is not None:
    image = img_file_buffer

if not image:
    uploaded_file = st.file_uploader("Upload an image",type=['png','jpg','jpeg','webp'])

    if uploaded_file:
        image = uploaded_file
        st.image(uploaded_file)
    
if image:
    faces = helper.get_face_from_upload(image)
    if len(faces)>1:
        st.write(str(len(faces)),'faces detected.')
    elif len(faces)==1:
        st.write('1 face detected.')
    else:
        st.write('Oops! No faces detected.')

    predit_btn = st.button('Predict mood',use_container_width=True)

    if predit_btn:
        helper.predict_faces(faces)