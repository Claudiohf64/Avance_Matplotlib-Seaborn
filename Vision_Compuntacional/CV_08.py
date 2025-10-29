import cv2

cap= cv2.VideoCapture('D:/Claudio\Vision_Compuntacional/run.mp4')
while True:
    ret, frame =  cap.read()
    if not ret:
        break
    frame= cv2.resize(frame, (640,480))
    cv2.rectangle(frame,(260,10),(10,190),(255,1,7),2)
    cv2.rectangle(frame,(260,260),(10,450),(255,1,7),2)
    cv2.rectangle(frame,(360,260),(620,450),(255,1,7),2)
    cv2.rectangle(frame,(360,10),(620,190),(255,1,7),2)

    cv2.line(frame,(30,20),(100,20),(0,255,7),2)
    cv2.line(frame,(240,20),(240,180),(0,255,7),2)

    cv2.line(frame,(30,440),(100,440),(0,255,7),2)
    cv2.line(frame,(240,270),(240,440),(0,255,7),2)

    cv2.line(frame,(520,440),(600,440),(0,255,7),2)
    cv2.line(frame,(380,270),(380,440),(0,255,7),2)

    cv2.line(frame,(520,20),(600,20),(0,255,7),2)
    cv2.line(frame,(380,20),(380,180),(0,255,7),2)

    cv2.imshow('Resultados', frame)
    tecla=cv2.waitKey(60)
    if tecla==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()