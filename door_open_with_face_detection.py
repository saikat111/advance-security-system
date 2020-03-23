import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from pyfirmata import Arduino
from time import sleep
from firebase import firebase
from datetime import date
board = Arduino('COM4')
firebase= firebase.FirebaseApplication("https://advance-security-system.firebaseio.com/advance-security-system", None)


def known():
    today = date.today()
    board.digital[13].write(1)
    firebase.put('/door', 'known', '1')
    firebase.put('/door', 'date', today)
    sleep(1)
    board.digital[13].write(0)


def unknown():
    board.digital[12].write(1)
    firebase.put('/door', 'unknown', '1')
    sleep(1)
    board.digital[12].write(0)


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

video_capture = cv2.VideoCapture(0)

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

while True:
    ret, img = video_capture.read()
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
    #img = img[:,:,::-1]

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        unknown()

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            known()

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
            #cv2.rectangle(img, (left - 25, top - 25), (right + 25, bottom + 25), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
            #cv2.rectangle(img, (left-10, bottom -10), (right+10, bottom+10), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 128, 0), 2)
            #cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting image


        cv2.imshow('Video', img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()




