import sys

packages = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("opencv", "cv2"),
    ("pillow", "PIL"),
    ("PyMuPDF", "fitz"),
    ("scikit-learn", "sklearn"),
    ("plotly", "plotly"),
    ("reportlab", "reportlab"),
    ("openpyxl", "openpyxl"),
    ("pytesseract", "pytesseract"),
    ("PaddlePaddle", "paddle"),
    ("PaddleOCR", "paddleocr")
]

all_passed = True
print("===================================================")
print("  ENVIRONMENT DIAGNOSTIC & DEPENDENCY CHECK")
print("===================================================")
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print("---------------------------------------------------")

for display_name, import_name in packages:
    try:
        __import__(import_name)
        print(f" [OK]      {display_name}")
    except ImportError as e:
        print(f" [MISSING] {display_name} (Error: {e})")
        all_passed = False

print("---------------------------------------------------")
if all_passed:
    print(" SUCCESS: All required packages are installed & working!")
    print("===================================================")
    sys.exit(0)
else:
    print(" WARNING: Some required packages are missing.")
    print(" Please run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    print("===================================================")
    sys.exit(1)
