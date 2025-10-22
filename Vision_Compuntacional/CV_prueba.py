import cv2
from ultralytics import YOLO

# --- Configuración Inicial ---
# Se utiliza un modelo de detección y se activa el seguimiento (tracker)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("peoplefull02.mp4")

linea_x = 320
person_count = 0
track_history = {} 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False) 

    H, W = frame.shape[:2]
    cv2.line(frame, (linea_x, 0), (linea_x, H), (0, 0, 255), 2)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
            if cls_id == 0 and conf > 0.5:
                x_center, y_center = int(box[0]), int(box[1])
                
                prev_x = track_history.get(track_id)

                cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(track_id), (x_center + 10, y_center - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if prev_x is not None:
                    if (prev_x < linea_x and x_center >= linea_x) or \
                       (prev_x > linea_x and x_center <= linea_x):
                        person_count += 1
                track_history[track_id] = x_center
    cv2.putText(frame, f'Personas cruzadas: {person_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Resultados', frame)
    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()