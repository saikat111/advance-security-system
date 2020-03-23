from tkinter import *
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from pyfirmata import Arduino
from time import sleep
from firebase import firebase
from datetime import date


top = Tk()

top.geometry("400x200")
top['bg'] = '#49A'
board = Arduino('COM4')
t = 0
firebase= firebase.FirebaseApplication("https://advance-security-system.firebaseio.com/advance-security-system", None)

def ledon():
    board.digital[11].write(1)
    sleep(1)
    board.digital[11].write(0)

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding




def fun():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frame1, (x, y), (x + y, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            global t
            t = t+1
            if t>10:
                ledon()
                t= t -t
        # cv2.drawContours(frame1,contours, -1, (0, 255, 0), 2)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def door():
    video_capture = cv2.VideoCapture(0)

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    while True:
        ret, img = video_capture.read()
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # img = img[:,:,::-1]

        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
                # cv2.rectangle(img, (left - 25, top - 25), (right + 25, bottom + 25), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                # cv2.rectangle(img, (left-10, bottom -10), (right+10, bottom+10), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 128, 0), 2)
                # cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

            # Display the resulting image

            cv2.imshow('Video', img)
            c = cv2.waitKey(1)
            if c == 27:
                break

    video_capture.release()
    cv2.destroyAllWindows()


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


def doordata():
    video_capture = cv2.VideoCapture(0)

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    while True:
        ret, img = video_capture.read()
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # img = img[:,:,::-1]

        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            unknown()
            name = "Unknown"

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
                # cv2.rectangle(img, (left - 25, top - 25), (right + 25, bottom + 25), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                # cv2.rectangle(img, (left-10, bottom -10), (right+10, bottom+10), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 128, 0), 2)
                # cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

            # Display the resulting image

            cv2.imshow('Video', img)
            c = cv2.waitKey(1)
            if c == 27:
                break

    video_capture.release()
    cv2.destroyAllWindows()


def motiondata():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frame1, (x, y), (x + y, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            global t
            t = t + 1
            if t > 0:
                database()
        # cv2.drawContours(frame1,contours, -1, (0, 255, 0), 2)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def database():
    today =date.today()
    firebase.put('/motion', 'times', str(t))
    firebase.put('/motion', 'date', today)


def detaction():
    video_capture = cv2.VideoCapture(0)

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    while True:
        ret, img = video_capture.read()
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # img = img[:,:,::-1]

        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
                # cv2.rectangle(img, (left - 25, top - 25), (right + 25, bottom + 25), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                # cv2.rectangle(img, (left-10, bottom -10), (right+10, bottom+10), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 128, 0), 2)
                # cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

            # Display the resulting image

            cv2.imshow('Video', img)
            c = cv2.waitKey(1)
            if c == 27:
                break

    video_capture.release()
    cv2.destroyAllWindows()

def car():
    car_cascade = cv2.CascadeClassifier('cars.xml')
    bus_cascade = cv2.CascadeClassifier('two_wheeler.xml')
    cap = cv2.VideoCapture('4.mp4')
    t = 0
    c = 0

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        car = car_cascade.detectMultiScale(gray, 1.1, 2)
        bus = bus_cascade.detectMultiScale(gray, 1.1, 2)

        for x, y, z, w in car:
            cv2.rectangle(img, (x, y), (x + z, y + w), (0, 255, 255), 2)
#            global t
            t = t + 1
            result = int(t/70)
            result2 = result
            if result2 <4:
                board.digital[10].write(0)
            if result2>5:
                board.digital[10].write(1)
                result2 = result2 - result2
            cv2.putText(img, "Car: {}".format(result), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            for m, n, o, p in bus:
                cv2.rectangle(img, (m, n), (m + o, n + p), (0, 255, 255), 2)
#                global c
                c = c + 1
                cv2.putText(img, "two_wheeler: {}".format(c), (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow("img", img)

        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()




b1 = Button(top, text="Motion", command=fun, activeforeground="red", activebackground="pink", pady=10)
b2 = Button(top, text="Door",command=door, activeforeground="blue", activebackground="pink", pady=10)
b3 = Button(top, text="Door with database",command=doordata, activeforeground="green", activebackground="pink", pady=10)
b4 = Button(top, text="Motion with database",command=motiondata, activeforeground="yellow", activebackground="pink", pady=10)
b5 = Button(top, text="person Detaction",command=detaction, activeforeground="yellow", activebackground="pink", pady=10)
b6 = Button(top, text="Road", command = car,activeforeground="yellow", activebackground="pink", pady=10)

b5.pack(anchor=CENTER)
b6.pack(anchor=CENTER)
b1.pack(anchor=CENTER)
b2.pack(anchor=CENTER)
b3.pack(anchor=CENTER)
b4.pack(anchor=CENTER)
b5.pack(anchor=CENTER)

top.mainloop()