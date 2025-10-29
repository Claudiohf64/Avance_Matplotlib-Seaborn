import cv2

cap= cv2.VideoCapture('D:/Claudio\Vision_Compuntacional/run.mp4')
while True:
    ret, frame =  cap.read()
    if not ret:
        break
    frame= cv2.resize(frame, (640,480))
    cv2.line(frame,(100,0),(100,480),(15,1,7),2)
    cv2.line(frame,(0,100),(frame.shape[1],100),(15,1,7),2)
    cv2.imshow('Resultados', frame)
    tecla=cv2.waitKey(60)
    if tecla==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()