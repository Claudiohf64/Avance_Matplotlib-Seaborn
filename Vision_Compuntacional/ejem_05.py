import cv2
img = cv2.imread('gato2.jpeg')
cv2.imshow('Original', img)
b, g, r = cv2.split(img)
cv2.imshow('Canal Azul', b)
cv2.imshow('Canal Verde', g)
cv2.imshow('Canal Rojo', r)
cv2.waitKey(0)
cv2.destroyAllWindows()