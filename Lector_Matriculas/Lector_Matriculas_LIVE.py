import cv2
import pytesseract
import re

# Configurar la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Expresión regular para matriculas europeas (4 dígitos seguidos de 3 letras)
#plate_pattern = re.compile(r'^\d{4}\s*[A-Za-z]{3}$')

# Expresión regular para matriculas (letra, 4 dígitos, 2 letras)
plate_pattern = re.compile(r'^[A-Za-z]-\d{4}-[A-Za-z]{2}$')


# Abrir la cámara del portátil (0 para la cámara por defecto)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Preprocesar la imagen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar bordes
    canny = cv2.Canny(gray, 50, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)

    # Encontrar contornos de posibles matrículas
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posible_matricula = []

    for contorno in contours:
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / float(h)

        if 2000 < area < 30000 and 2 < aspect_ratio < 5:
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            if len(approx) == 4:
                posible_matricula.append((x, y, w, h))

    # Extraer y dibujar el texto de las matrículas
    for matricula in posible_matricula:
        x, y, w, h = matricula
        roi = gray[y:y+h, x:x+w]

        # Mejora del ROI para reconocimiento
        roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(roi, config='--psm 8').strip()

        # Dibujar todas las posibles matrículas
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Verificar si el texto cumple con el patrón de la matrícula
        if plate_pattern.match(text):
            print(f"Matrícula detectada: {text}")

    # Mostrar el resultado
    cv2.imshow('Lectura de matriculas', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()

