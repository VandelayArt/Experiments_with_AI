# Import cv2 library
import cv2

from random import randrange

# Load trained data from opencv2's sample library (haar cascade algorithm)
face_data = cv2.CascadeClassifier('data/haarcascade_frontalfaces_default.xml')


# Store the image into a variable
img1 = cv2.imread('img/img-2.jpg')

# Turn the image into a black and white image (grayscaling)
gs1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# Detecting the face
face_coordinates = face_data.detectMultiScale(gs1)
print(face_coordinates)

# Drawing boxes around the face using the coordinates variable instead of hardcoding the x, y values
    # parameters in order: image, top left x/y, bottom right x/y, color of border, thickness of border
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img1, (x,y), (x+w, y+w), (randrange(256), randrange(256), randrange(256)), 3)


 # Showing the image
cv2.imshow("Face Detector App", img1)
# Holding the window open while showing the image
cv2.waitKey()

print("Code Completed")