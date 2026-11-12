import cv2
from PIL import Image
import numpy as np

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
img = cv2.imread(img_path)
print(f"Shape: {img.shape}") # (height, width, channels)

# Also check EXIF orientation in PIL
pil_img = Image.open(img_path)
print(f"PIL size: {pil_img.size}")
exif = pil_img.getexif()
print(f"EXIF: {dict(exif) if exif else 'None'}")
