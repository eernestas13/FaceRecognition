import math

import cv2
import numpy as np
import face_recognition
import os
from flask import Flask, Response, render_template
from flask_socketio import SocketIO
# import SocketIO
# import flask_socketio

import socket

hostName = socket.gethostname()
ipAddress = socket.gethostbyname(hostName)
app = Flask(__name__)
socketIOApp = SocketIO(app)
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)

print(ipAddress)
# Creates a Similarity Score through a %
def faceSimilarity(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

# Append Images and Names to Arrays
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Returns an Array of Image Encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Finished!')

# Opens Camera
cap = cv2.VideoCapture(0)


def getframes():
    while True:
        # If Video Capture is succesful
        success, img = cap.read()
        # Resizing image to make face recognition faster
        imgS = cv2.resize(img, (0, 0,), None, 0.25, 0.25)
        # Color Scheme conversion from BGR to RGB
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        # Loop of face encodings and compares them
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            # If a face matches within the threshold, creates a green box
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                likelyMatch = round(faceSimilarity(faceDis[matchIndex])*100)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name+" "+str(likelyMatch)+"%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # If a face doesn't match within the threshold, creates a blue box
            else:
                name = 'unknown'
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

# Camera feed on the Homepage
@app.route('/video_feed')
def video_feed():
    return Response(getframes(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Homepage of localhost
@app.route('/')
def index():
    return render_template('index.html')


def run():
    socketIOApp.run(app)


if __name__ == '__main__':
    socketIOApp.run(app)
