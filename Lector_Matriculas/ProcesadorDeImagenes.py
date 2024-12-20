import cv2
import pytesseract

# Configurar la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('CocheBeige.jpg')


# Transformar color de imagen de BGR a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3, 3))
canny = cv2.Canny(gray, 150, 200)
canny = cv2.dilate(canny, None, iterations=1)


# Dibujar contornos
contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Desechar contornos no deseados
for contorno in contours:
    area = cv2.contourArea(contorno)
    x, y, w, h = cv2.boundingRect(contorno)
    epsilon = 0.09 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)
    if len(approx) == 4 and area > 9000:
        print('area', area)
        cv2.drawContours(image, [contorno], 0, (0, 255, 0), 2)



cv2.imshow('Image', image)
cv2.imshow('Canny', canny)
cv2.moveWindow('Image', 45, 10)
cv2.waitKey(0)

