# from pydub import AudioSegment

# final = AudioSegment.empty()

# for i in range(1, 10):
#     final += AudioSegment.from_mp3(f"{i}.mp3")

# final.export("final_quiz_intro.mp3", format="mp3")

import easyocr
import cv2
import re

image_path = "image.png"   

# Load image
image = cv2.imread(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Perform OCR
results = reader.readtext(image)

# Extract text
detected_texts = [res[1] for res in results]

print("Detected Text:", detected_texts)

# Find MFD date pattern (dd/mm/yy or dd/mm/yyyy)
date_pattern = r'\b\d{2}/\d{2}/\d{2,4}\b'

mfd_date = None

for text in detected_texts:
    match = re.search(date_pattern, text)
    if match:
        mfd_date = match.group()
        break

if mfd_date:
    print("Extracted MFD Date:", mfd_date)
else:
    print("MFD Date not found")