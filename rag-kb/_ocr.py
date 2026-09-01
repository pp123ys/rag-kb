import sys
from paddleocr import PaddleOCR

img = sys.argv[1]
ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
result = ocr.ocr(img, cls=True)
for line in result or []:
    for box, text in line:
        print(text)
