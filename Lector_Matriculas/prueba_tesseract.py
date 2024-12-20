import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
image = cv2.imread('PruebaTesseract2.png')

text = pytesseract.image_to_string(image)
print("Texto de la imagen:",text)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()