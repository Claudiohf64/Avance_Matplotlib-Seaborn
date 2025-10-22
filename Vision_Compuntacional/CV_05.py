import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("run.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640, 480))
    resultados = model(frame, verbose=False)
    
    cv2.line(frame, (550,180), (550,30),(0,0,255),2)

    cv2.line(frame, (550,260), (550,420),(0,0 ,255),2)
    
    cv2.line(frame, (100,180), (100,30),(0,255,0),2)

    cv2.line(frame, (100,260), (100,420),(0,255 ,0),2)

    cv2.imshow('Resultados', frame)
    tecla = cv2.waitKey(50)
    if tecla == ord('q'):
        break
    
cap.release()