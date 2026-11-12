import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import re
import numpy as np
from PIL import Image

from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT
from ocr.preprocessing import (
    convert_to_opencv,
    correct_document_perspective,
    remove_shadows_and_enhance
)
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()

# Test sample files
sample_files = [
    ("IMG_4867 (Bhavana)", r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"),
    ("IMG_4839 (Parthu)", r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg"),
    ("IMG_5774 (Bhavana)", r"data\uploads\09b450b9_IMG_5774 01.PNG"),
    ("WhatsApp 03 (Raghuvardhan)", r"data\uploads\25b29d44_WhatsApp Image 03.jpeg"),
    ("IMG_5827 (Neethu)", r"data\uploads\e65f4245_IMG_5827.JPG 002.jpeg"),
    ("IMG_5829 (Prudhvi)", r"data\uploads\9fb5d70e_IMG_5829.JPG 003.jpeg"),
    ("Image 004 (Rishika)", r"data\uploads\4ddfea8c_Image 004.jpeg")
]

print("=== VERIFYING SAMPLE FILES EXISTENCE ===")
for name, path in sample_files:
    exists = os.path.exists(path)
    print(f"  {name:30s}: {'EXISTS' if exists else 'MISSING'} ({path})")
