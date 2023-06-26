# 필요한 모듈 임포트
import cv2 as cv
import numpy as np
import dlib
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import random

# 학습 모델 로드
emotion_model_path = 'model/emotion_model.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# 감정 레이블 정의
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
FAKE_EMOTIONS = ["happy", "surprised", "neutral"]
EMOTION_REAL = ["angry", "sad", "scared", "neutral"]

# Dlib 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠 영상 로드
cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# 감정 레이블을 추적하기 위한 변수 초기화
prev_label = None
real_label = None

while True:
    ret, frame = cap.read()
    
    # grayscale 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    faces = detector(gray)
    
    # 감지된 얼굴만큼 반복
    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        
        # ROI 추출 및 전처리
        roi = gray[y:y + h, x:x + w]
        roi = cv.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # 감정 예측
        preds = emotion_classifier.predict(roi)[0]
        
        # fake emotion에 해당하는 결과만 추출
        fake_preds = [preds[EMOTIONS.index(emotion)] for emotion in FAKE_EMOTIONS]
        label = FAKE_EMOTIONS[np.argmax(fake_preds)]
        
        # 감정 레이블이 변경되었는지 확인
        if label != prev_label:
            # 실제 감정 레이블을 랜덤하게 변경
            real_label = random.choice(EMOTION_REAL)
            
        # 레이블을 프레임에 그리기
        cv.putText(frame, "You outside the mask: " + label, (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 'real appearance'를 아래쪽에 추가
        cv.putText(frame, "You in the mask: " + real_label, (x, y + h + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
        
        # 이전 레이블 업데이트
        prev_label = label
        
    # 결과 프레임
    cv.imshow('Emotion Recognition', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 리소스 해제
cap.release()
cv.destroyAllWindows()
