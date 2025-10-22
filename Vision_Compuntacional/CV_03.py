import cv2
# Para Personas
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
cap = cv2.VideoCapture("smile.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(gray, (640, 480))

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=10)
    eyes = eye_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=10)
    mouth = mouth_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=50)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for x, y, w, h in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for x, y, w, h in mouth:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Resultado", frame)
    tecla=cv2.waitKey(50)
    if tecla == ord('q'):
        break
cap.release()
