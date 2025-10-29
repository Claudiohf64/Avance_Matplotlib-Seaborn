import cv2
import mediapipe as mp

# Inicializamos los módulos de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Índices de las puntas de los dedos
FINGER_TIPS = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular, meñique

logo = cv2.imread("D:/Claudio/Vision_Compuntacional/alex.jpg", cv2.IMREAD_UNCHANGED)
logo_resized = cv2.resize(logo, (160, 168))
# Captura de cámara
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hands_open = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Comprobamos si la mano está abierta
                finger_open_count = 0
                h, w, _ = frame.shape
                for tip_idx in FINGER_TIPS[1:]:  # índice, medio, anular, meñique
                    tip_y = hand_landmarks.landmark[tip_idx].y * h
                    pip_y = hand_landmarks.landmark[tip_idx - 2].y * h
                    if tip_y < pip_y:
                        finger_open_count += 1
                # Pulgar (horizontal)
                thumb_tip_x = hand_landmarks.landmark[FINGER_TIPS[0]].x * w
                thumb_ip_x = hand_landmarks.landmark[3].x * w
                if abs(thumb_tip_x - thumb_ip_x) > 20:
                    finger_open_count += 1

                if finger_open_count == 1:
                    hands_open = True

        # Si hay mano abierta, pasamos a escala de grises
        if hands_open:
            x_offset, y_offset = 100, 150  # posición en el frame
            frame[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1]] = logo_resized

        # Agregar título grande
        text = "tengo contacto con dios" if hands_open else ""
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 5, cv2.LINE_8)  # Tamaño grande, color verde, grosor 5

        # Mostrar frame
        cv2.imshow('Detección de Manos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
