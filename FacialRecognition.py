'''
Created on Aug 20, 2015

@author: Shuyi Wang
'''

if __name__ == '__main__':
    pass

import cv2
import sys

cascPath = sys.argv[1]
cameraIndex = 0
if len(sys.argv) > 2:
    cameraIndex = int(sys.argv[2])
print("Cascade file will be loaded from :" + cascPath)
print("Camera index: " + str(cameraIndex))
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(cameraIndex)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE  # New constant for opencv 3.0.0
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
