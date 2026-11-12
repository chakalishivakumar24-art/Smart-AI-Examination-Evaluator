import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()

# Let's inspect the marks table on warped_09b450b9_IMG_5774 01.PNG.jpg
img_path = r"scratch\warped_09b450b9_IMG_5774 01.PNG.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# On 1600 x 2200, the table region is approximately y=1000 to y=2050, x=50 to x=1550
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

# Detect horizontal lines
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

# Detect vertical lines
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

table_mask = cv2.add(h_lines, v_lines)
cv2.imwrite(r"scratch\test_table_mask.jpg", table_mask)

# Find contours of horizontal lines to find row dividers
h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h_y_coords = []
for c in h_contours:
    rx, ry, rw, rh = cv2.boundingRect(c)
    if rw > 300 and ry > 900: # only horizontal table dividers
        h_y_coords.append(ry)
h_y_coords = sorted(list(set(h_y_coords)))

# Cluster horizontal lines within 15px
clustered_y = []
for y in h_y_coords:
    if not clustered_y or abs(y - clustered_y[-1]) > 15:
        clustered_y.append(y)

print(f"Found {len(clustered_y)} table row dividers:")
for idx, y in enumerate(clustered_y):
    print(f"  Row line {idx}: y = {y}")

# Find contours of vertical lines to find column dividers
v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
v_x_coords = []
for c in v_contours:
    rx, ry, rw, rh = cv2.boundingRect(c)
    if rh > 300 and ry > 900: # only vertical table dividers
        v_x_coords.append(rx)
v_x_coords = sorted(list(set(v_x_coords)))

clustered_x = []
for x in v_x_coords:
    if not clustered_x or abs(x - clustered_x[-1]) > 15:
        clustered_x.append(x)

print(f"\nFound {len(clustered_x)} table column dividers:")
for idx, x in enumerate(clustered_x):
    print(f"  Col line {idx}: x = {x}")
