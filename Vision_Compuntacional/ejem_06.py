import cv2
# Para Gatos
cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
)
img = cv2.imread("gato2.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cats = cat_cascade.detectMultiScale(gray, 1.3, 5)
for x, y, w, h in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Resultado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()