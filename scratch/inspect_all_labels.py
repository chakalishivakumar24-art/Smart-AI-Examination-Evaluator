import cv2
import pytesseract
from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\test_normalized_page.jpg"
bgr = cv2.imread(img_path)
h, w = bgr.shape[:2]
if w != NORMALIZED_WIDTH or h != NORMALIZED_HEIGHT:
    bgr = cv2.resize(bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

data = pytesseract.image_to_data(clahe, output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.astype(str).str.strip() != '')]

for _, r in data.iterrows():
    txt = str(r['text']).strip()
    x, y, w, h = int(r['left']), int(r['top']), int(r['width']), int(r['height'])
    conf = float(r['conf'])
    if any(k in txt.lower() for k in ["enroll", "reg", "date", "sem", "title", "branch", "dept", "sec", "class", "test", "serial", "code", "sheet", "recheck", "remark"]):
        print(f"  {txt:25s}: x={x:4d}, y={y:4d}, w={w:4d}, h={h:4d} (Conf={conf:.1f}%)")
