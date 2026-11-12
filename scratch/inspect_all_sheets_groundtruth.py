import sys
sys.path.insert(0, ".")
import cv2
import os

files = {
    "sheet1": r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg",
    "sheet2": r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg",
    "sheet3": r"data\uploads\09b450b9_IMG_5774 01.PNG",
    "sheet4": r"data\uploads\25b29d44_WhatsApp Image 03.jpeg",
    "sheet5": r"data\uploads\d9c1ef5e_IMG_5827.JPG 002.jpeg"
}

for k, p in files.items():
    if os.path.exists(p):
        img = cv2.imread(p)
        print(f"{k}: {p} -> {img.shape if img is not None else 'None'}")
    else:
        print(f"{k}: not found")
