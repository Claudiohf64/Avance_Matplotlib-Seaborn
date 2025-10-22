import cv2

# Para Personas
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
)
cap = cv2.VideoCapture("cats.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames = cv2.resize(gray, (640, 480))

    faces = face_cascade.detectMultiScale(frames, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Resultado",frames)
    tecla=cv2.waitKey(50)
    if tecla == ord('q'):
        break
cap.release()