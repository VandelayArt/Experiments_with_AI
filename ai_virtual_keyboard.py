# Importing the OpenCV library for translating the image to an array and drawing on it
import cv2

# Import cvzone's Hand Tracker, to register handmovements 
from cvzone.HandTrackingModule import HandDetector

# Importing the randrange to randomly switch between colours
from random import randrange

# Importing a sleep fucntion to delay the time between selected keys
from time import sleep

# Loading the webcam and setting it's resolution to 1280x720
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)


# Loading cvzone's HandDetector and setting it's 'detection confidence'
detector = HandDetector(detectionCon=0.8)

# A list of keyboard touches
touch_keys = [
    ["@","Q","W","E","R","T","Y","U","I","O","P","*"],
    [">","A","S","D","F","G","H","J","K","L","+","/"],
    ["<","Z","C","V","B","N","M",",",".","-","_",":"]
]

# Final Text to display selected keys
final_text = ""


# A function that draws buttons on the screen
def drawAll(img, button_list):
    for button in button_list:
        x,y = button.pos
        w,h = button.size
        cv2.rectangle(img, button.pos, (x+w, y+h), (255,255,0), cv2.BORDER_CONSTANT)
        cv2.putText(img, button.text, (x, y+65), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 5)
    # Return the manipulated img ---OUTSIDE THE LOOP---
    return img


# A class to define different variables for each button automatically
class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text


# Creating a list of buttons to iterate through based on the "touch_key" list 
button_list = []
for i in range(len(touch_keys)):
    # A loop that iterates through the values inside of each value of the list
    for x,touch_key in enumerate(touch_keys[i]):
        # Adding defined Buttons (from class) to our "button_list"
        button_list.append(Button([100 * x + 50, 100 * i + 50], touch_key))



# The loop to run our webcam feed and it's script
while True:
    # Reading the webcam frame
    success, img = cap.read()

    # Only need to implement the following flip function if your webcam mirrors your image automatically
    img = cv2.flip(img, 1)

    # Using the HandDetector to identify the hands
    hands, img = detector.findHands(img)
    
    # Draw the buttons on the screen
    drawAll(img, button_list)    


    # Loading hands
    if hands:
    # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"
            
            fingers2 = detector.fingersUp(hand2)
            
    
        # Loop that will analyze each button and recognize the keys you press
        for button in button_list:
            x, y = button.pos
            w, h = button.size

            # If statement for index finger to be positioned inside of button area
            if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y+h:
                cv2.rectangle(img, button.pos, (x+w, y+h), (randrange(256), randrange(256), randrange(256)), cv2.BORDER_CONSTANT)
                cv2.putText(img, button.text, (x, y+65), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 5)

                # Calculating distance between index and middle finger     
                l,_,_ = detector.findDistance(lmList1[8], lmList1[12], img)
                
                # If statement for when fingers are close enough together
                if l < 40:
                    cv2.rectangle(img, button.pos, (x+w, y+h), (randrange(256), randrange(256), randrange(256)), cv2.FILLED)
                    cv2.putText(img, button.text, (x, y+65), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 5)

                    # Add selected key to the displayed text
                    final_text += button.text
                    # Buffering so that key doesn`t get pressed each milisecond
                    sleep(1)

    # Displaying typed phrase
    cv2.rectangle(img, (25,550), (1000, 650), (randrange(256), randrange(256), randrange(256)), cv2.FILLED)
    cv2.putText(img, final_text, (30, 625), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,255), 5)

    # Show the image
    cv2.imshow("Image", img)
    

    # Closing the program
    key = cv2.waitKey(1)

    # Close the program if "Q" is pressed
    if key==81 or key==113:
        break

# Release the webcam of it's duties
cap.release()