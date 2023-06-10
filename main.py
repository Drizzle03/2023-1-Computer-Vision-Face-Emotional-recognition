import cv2, dlib, sys 
import numpy as np
cap = cv2.VideoCapture('girl.mp4')

scaler = 0.3

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
predictor_file = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)


while True:
    ret, img = cap.read()'shape_predictor_68_face_landmarks.dat'
    if not ret:
        break

    #img size 조정
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler )))
    ori = img.copy()

    cv2.imshow('img',img) #img라는 이름의 윈도우에 img 띄우기
    cv2.waitKey(1) #딜레이 1밀리세컨드