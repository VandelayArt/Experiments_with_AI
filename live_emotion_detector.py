# OpenCV library to use for image recognition
import cv2

# Importing a trained model to analyze the frame recorded by the webcam
from deepface import DeepFace

# Importing randrange, used here to randomly switch between colours to display the rectangle and text
from random import randrange

# Loading the trained data and storing it into a variable
facecascade = cv2.CascadeClassifier('data/haarcascade_frontalfaces_default.xml')

# Loading the webcam and storing it into a variable
cap = cv2.VideoCapture(0)

# Setting a condition to run the script
while True:
    # Reading the live frame
    ret, frame = cap.read()

    # Analyzing the frame using DeepFace module and specifying what to analyze
        #Options are gender, dominant emotion, dominant race and age
    result = DeepFace.analyze(frame,actions=['dominant_emotion', 'gender', 'dominant_race',], enforce_detection=False)
    
    # Storing the dominant_emotion in a callable variable
    emotion = result['dominant_emotion']
    
    gender = result['gender']
    race = result['dominant_race']

    # Combining all required values from the analysis dictionary
    combined_result = str(emotion + "," + gender + "," + race) 
    
    
    # Grayscaling the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting the face/faces
    faces = facecascade.detectMultiScale(gray,1.1,4)

    # Assigning a for loop to define coordinates on which to draw the outline
        # of the faces. Randrange is fun to rapidly switch between random colors
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)

    # Setting a font for the text
    font = cv2.FONT_HERSHEY_COMPLEX

    
    # Defining the text to show on screen by a series of if statements
        # If happy
    if emotion == "happy":    
        cv2.putText(frame, combined_result, (100,350), font, 1,
                    (randrange(256), randrange(256), randrange(256)),
                    2, cv2.LINE_4
                )
        # If sad
    elif emotion == "sad":
        cv2.putText(frame, combined_result, (100,350), font, 1,
                    (randrange(256), randrange(256), randrange(256)),
                    2, cv2.LINE_4
                )
        # If angry
    elif emotion == "angry":
        cv2.putText(frame, combined_result, (100,350), font, 1,
                    (randrange(256), randrange(256), randrange(256)),
                    2, cv2.LINE_4
                )
        # If surprised        
    elif emotion == "surprise":
        cv2.putText(frame, combined_result, (100,350), font, 1,
                    (randrange(256), randrange(256), randrange(256)),
                    2, cv2.LINE_4
                )
        # If no specific emotion is recognized
    elif emotion == "neutral":
        cv2.putText(frame, combined_result, (100,350), font, 1,
                    (randrange(256), randrange(256), randrange(256)),
                    2, cv2.LINE_4
                )
    
    # Showing the frame
    cv2.imshow('Emotion Detector', frame)

    # Storing the waitKey(holds open the program until a button is pressed) in a variable
    key = cv2.waitKey(1)

    # Close the program if "Q" is pressed
    if key==81 or key==113:
        break

# Release the webcam of its duties
cap.release()