import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import dlib
from imutils import face_utils


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp20/weights/last.pt', force_reload=True)
vi='http://192.168.0.108/4747/video'
vi2=r'E:\Huawei AI\Projek akhir\SleepyDetect\slsys\data\VID_20211230_170913.mp4'
vi3='idm2-low.mp4'
cap=cv2.VideoCapture(vi)
detector=dlib.get_frontal_face_detector()
# detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
prediksi=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep=0
drowsy=0
active=0
status=""
color=(0,0,0)
YAWN_THRESH = 20

def compute(pta,ptb):
    dist=np.linalg.norm(pta-ptb)
    return dist

def blinked(a,b,c,d,e,f):
    up=compute(b,d) + compute(c,e)
    down=compute(a,f)
    ratio=up/(2.0*down)
    
    if(ratio > 0.25):
        return 2
    elif(ratio >0.21 and ratio <=0.25):
        return 1
    else:
        return 0

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance
    
while True:
    ret,frame=cap.read()
    results=model(frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        
        faceframe=frame.copy()
        cv2.rectangle(faceframe, (x1,y1), (x2,y2), (0,255,0),2)
        landmarks=prediksi(gray,face)
        landmarks=face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        distance=lip_distance(landmarks)
        lip = landmarks[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status="SLEEPING !!!"
                color = (255,0,0)

        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status="Drowsy !"
                color = (0,0,255)

        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="Active :)"
                color = (0,255,0)

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(faceframe, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", np.squeeze(results.render()))
    cv2.imshow("Result of detector", faceframe)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
        
        