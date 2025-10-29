import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

emociones= ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model= load_model('D:\Claudio\RNA_CONV.h5')
img= cv2.imread('D:/Claudio/Vision_Compuntacional/archive(6)/extras/disgust/182000000.jpg', cv2.IMREAD_GRAYSCALE)
img_resized= cv2.resize(img, (48,48))
img_input= img_resized.reshape(1,48,48,1)
predict=model.predict(img_input)
label_predic= np.argmax(predict)
labels=emociones[label_predic]
 