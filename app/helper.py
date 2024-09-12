import cv2
import numpy as np
import pickle as pkl

emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


model = pkl.load(open('../model/face_rec_model.pkl','rb'))

# Load the pre-trained Haar Cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_from_upload(img):
    output = []
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes,1)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces and process each
    for (x, y, w, h) in faces:
        # Crop the face from the grayscale image
        face = gray_image[y:y+h, x:x+w]

        # Resize the face to 48x48 pixels
        resized_face = cv2.resize(face, (48, 48))
        cv2.waitKey(0)
        output.append(resized_face)
    cv2.destroyAllWindows()
    return output

def predict(image):
    # Normalize the image and reshape to the input shape of the model
    resized_image = image / 255.0
    resized_image = resized_image.reshape(1, 48, 48, 1)

    # Predict the emotion
    emotion_prediction = model.predict(resized_image)
    max_index = np.argmax(emotion_prediction)

    # Display the predicted emotion
    predicted_emotion = emotion_map[max_index]
    return predicted_emotion