import cv2

from random import randrange

# Load the trained data using the CascadeClassifier
face_data = cv2.CascadeClassifier('data/haarcascade_frontalfaces_default.xml')

# To capture video from camera
webcam = cv2.VideoCapture(0)

while True:
    # Reading the live frame
    successful_frame, frame = webcam.read()

    # Grayscaling the frames
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_data.detectMultiScale(gs_frame)


# Drawing boxes around the face using the coordinates variable instead of hardcoding the x, y values
    # parameters in order: image, top left x/y, bottom right x/y, color of border, thickness of border
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(gs_frame, (x,y), (x+w, y+w), (randrange(256), randrange(256), randrange(256)), 3)


    cv2.imshow('Live Face Detector', gs_frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()