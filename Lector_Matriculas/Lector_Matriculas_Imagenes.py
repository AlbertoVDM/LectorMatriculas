import cv2
import pytesseract

# Configurar la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar la imagen
image = cv2.imread('CocheBeige.jpg')

# Transformar color de imagen de BGR a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

#Detectar bordes en imagen blanco/negro
canny = cv2.Canny(gray, 50, 150)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)

# Encontrar contornos de posibles matrículas
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
posibles_matriculas = []

for contorno in contours:
    area = cv2.contourArea(contorno)
    x, y, w, h = cv2.boundingRect(contorno)
    aspect_ratio = w / float(h)

    if 2000 < area < 30000 and 2 < aspect_ratio < 5:
        epsilon = 0.02 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)
        if len(approx) == 4:
            posibles_matriculas.append((x, y, w, h))

# Extraer y dibujar el texto de las matrículas
for matricula in posibles_matriculas:
    x, y, w, h = matricula
    roi = image[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi, config='--psm 8').strip()
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar el resultado
cv2.imshow('Lector de matriculas', image)
cv2.waitKey(0)