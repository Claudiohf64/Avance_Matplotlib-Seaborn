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

    cv2.line(frame,(30,60),(100,20),(0,255,7),2)
    cv2.line(frame,(240,60),(175,20),(0,255,7),2)

    cv2.line(frame,(30,130),(100,170),(0,255,7),2)
    cv2.line(frame,(240,130),(175,170),(0,255,7),2)


    cv2.line(frame,(30,330),(100,280),(0,255,7),2)
    cv2.line(frame,(240,330),(175,280),(0,255,7),2)

    cv2.line(frame,(240,380),(175,430),(0,255,7),2)
    cv2.line(frame,(30,380),(100,430),(0,255,7),2)


    cv2.line(frame,(380,330),(450,280),(0,255,7),2)
    cv2.line(frame,(600,330),(540,280),(0,255,7),2)

    cv2.line(frame,(380,380),(450,430),(0,255,7),2)
    cv2.line(frame,(600,380),(540,430),(0,255,7),2)


    cv2.line(frame,(380,60),(450,20),(0,255,7),2)
    cv2.line(frame,(600,60),(540,20),(0,255,7),2)

    cv2.line(frame,(380,130),(450,170),(0,255,7),2)
    cv2.line(frame,(600,130),(540,170),(0,255,7),2)

    cv2.imshow('Resultados', frame)
    tecla=cv2.waitKey(60)
    if tecla==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()