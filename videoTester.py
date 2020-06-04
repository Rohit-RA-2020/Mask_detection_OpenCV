import os
import cv2
import numpy as np
import faceRecognition as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('F://face//trainingData.yml')#Load saved training data

name = {0 : "Mask on", 1:"Without mask"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        if label == 0:
          fr.draw_rect(test_img, face, 0)
        
        elif label == 1:
          fr.draw_rect(test_img, face, 1)
        
        predicted_name=name[label]
        #if confidence < 39:#If confidence less than 37 then don't print predicted face text on screen
        fr.put_text(test_img,predicted_name,x,y)



    for (x,y,w,h) in faces_detected:
      if label == '0':
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=1)
      elif label == '1':
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=2)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ',resized_img)
    cv2.waitKey(10)


    


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
