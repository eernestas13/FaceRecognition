import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Images/elonMusk.png')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgElon2 = face_recognition.load_image_file('ImagesTest/elonMusk2.png')
imgElon2 = cv2.cvtColor(imgElon2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc2 = face_recognition.face_locations(imgElon2)[0]
encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
cv2.rectangle(imgElon2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeElon2)
faceDis = face_recognition.face_distance([encodeElon],encodeElon2)
print(results,faceDis)
cv2.putText(imgElon,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Musk2', imgElon2)

cv2.waitKey(0)