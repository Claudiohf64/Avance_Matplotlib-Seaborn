import cv2
import mediapipe as mp
mp_face_mesh= mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh(max_num_faces=1)
EYE_LEFT=[33,133]
EYE_RIGHT=[362,263]
MOUTH_L=[61,40,91,37,84,0,17]
MOUTH_R=[267,314,270,321,409,375,291]
cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()
    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results= face_mesh.process(frame_rgb)
    for face_landmarks in results.multi_face_landmarks:
        h,w,_= frame.shape
        eyeL_left= (int(face_landmarks.landmark[EYE_LEFT[0]].x*w), 
                         int(face_landmarks.landmark[EYE_LEFT[0]].y*h))
        eyeL_right= (int(face_landmarks.landmark[EYE_LEFT[1]].x*w), 
                         int(face_landmarks.landmark[EYE_LEFT[1]].y*h))
        
        eyeR_left= (int(face_landmarks.landmark[EYE_RIGHT[0]].x*w), 
                         int(face_landmarks.landmark[EYE_RIGHT[0]].y*h))
        eyeR_right= (int(face_landmarks.landmark[EYE_RIGHT[1]].x*w), 
                         int(face_landmarks.landmark[EYE_RIGHT[1]].y*h))
        cv2.circle(frame, eyeL_left, 3, (0,0,255), -1)
        cv2.circle(frame, eyeL_right, 3, (0,0,255), -1)
        cv2.circle(frame, eyeR_left, 3, (0,255,0), -1)
        cv2.circle(frame, eyeR_right, 3, (0,255,0), -1)

    for mouth_landmarks in results.multi_face_landmarks:
        h,w,_= frame.shape
        mouthL_01= (int(face_landmarks.landmark[MOUTH_L[0]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[0]].y*h))
        mouthL_02= (int(face_landmarks.landmark[MOUTH_L[1]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[1]].y*h))
        mouthL_03= (int(face_landmarks.landmark[MOUTH_L[2]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[2]].y*h))                                                                                                                            
        mouthL_04= (int(face_landmarks.landmark[MOUTH_L[3]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[3]].y*h))
        mouthL_05= (int(face_landmarks.landmark[MOUTH_L[4]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[4]].y*h))
        mouthL_06= (int(face_landmarks.landmark[MOUTH_L[5]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[5]].y*h))
        mouthL_07= (int(face_landmarks.landmark[MOUTH_L[6]].x*w), 
                         int(face_landmarks.landmark[MOUTH_L[6]].y*h))
        
        mouthR_01= (int(face_landmarks.landmark[MOUTH_R[0]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[0]].y*h))
        mouthR_02= (int(face_landmarks.landmark[MOUTH_R[1]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[1]].y*h))
        mouthR_03= (int(face_landmarks.landmark[MOUTH_R[2]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[2]].y*h))
        mouthR_04= (int(face_landmarks.landmark[MOUTH_R[3]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[3]].y*h))
        mouthR_05= (int(face_landmarks.landmark[MOUTH_R[4]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[4]].y*h))
        mouthR_06= (int(face_landmarks.landmark[MOUTH_R[5]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[5]].y*h))
        mouthR_07= (int(face_landmarks.landmark[MOUTH_R[6]].x*w), 
                         int(face_landmarks.landmark[MOUTH_R[6]].y*h))
        
        cv2.circle(frame, mouthL_01, 3, (255,0,255), -1)
        cv2.circle(frame, mouthL_02, 3, (255,0,255), -1)
        cv2.circle(frame, mouthL_03, 3, (255,0,255), -1)
        cv2.circle(frame, mouthL_04, 3, (255,0,255), -1)
        cv2.circle(frame, mouthL_05, 3, (255,0,255), -1)
        cv2.circle(frame, mouthL_06, 3, (255,255,255), -1)
        cv2.circle(frame, mouthL_07, 3, (255,255,255), -1)

        cv2.circle(frame, mouthR_01, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_02, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_03, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_04, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_05, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_06, 3, (255,255,0), -1)
        cv2.circle(frame, mouthR_07, 3, (255,255,0), -1)
    cv2.imshow('Resultados: ', frame)
    tecla= cv2.waitKey(1)
    if tecla==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()