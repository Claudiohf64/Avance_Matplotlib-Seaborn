import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("run.mp4")
contador = 0
linea_x = 320
personas_previas = []
velocidad_lenta = 50

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    resultados = model(frame, verbose=False)
    cv2.line(frame, (linea_x, 0), (linea_x, frame.shape[0]), (0, 0, 255), 2)
    personas_nuevas=[]

    for r in resultados:
        for box in r.boxes:
            if int(box.cls[0])==0:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                centro_x= (x1+x2)//2
                centro_y=(y1+y2)//2
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.circle(frame, (centro_x,centro_y),5,(255,0,0),-1)
                personas_nuevas.append((centro_x,centro_y))
                for (prev_x,prev_y) in personas_previas:
                    if prev_x < linea_x <= centro_x:
                        contador +=1
                        print(contador)
                        break
    personas_previas = personas_nuevas.copy()
    cv2.putText(frame,f'Contador {contador}', (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    cv2.imshow("Resultados", frame)
    tecla = cv2.waitKey(60)

    if tecla == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
