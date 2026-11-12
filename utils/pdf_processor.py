try:
    import pymupdf as fitz
except ImportError:
    import fitz
from PIL import Image, ImageOps
import io
import os
import uuid
from typing import List, Tuple

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def save_uploaded_file(file_bytes: bytes, original_filename: str) -> Tuple[str, str]:
    """
    Validates uploaded file, generates a unique filename to prevent collisions,
    saves the file to data/uploads/, and returns (saved_path, unique_filename).
    """
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise ValueError("Invalid file format. Only PDF, JPG, JPEG, and PNG files are supported.")

    unique_filename = f"{uuid.uuid4().hex[:8]}_{original_filename}"
    saved_path = os.path.join(UPLOADS_DIR, unique_filename)

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    return saved_path, unique_filename

def convert_bytes_to_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    """
    Converts uploaded file bytes (PDF/PNG/JPG/JPEG) into a list of PIL Images using PyMuPDF.
    Automatically applies EXIF transpose to respect smartphone/camera orientation.
    """
    filename_lower = filename.lower()
    images = []

    if filename_lower.endswith('.pdf'):
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(pdf_doc) == 0:
                raise ValueError("The uploaded PDF document contains 0 pages.")

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(dpi=150)
                raw_img = Image.open(io.BytesIO(pix.tobytes("png")))
                img = ImageOps.exif_transpose(raw_img).convert("RGB")
                images.append(img)
            pdf_doc.close()
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable PDF file: {e}")
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        try:
            raw_img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            images.append(img)
        except Exception as e:
            raise ValueError(f"Unreadable image file: {e}")
    else:
        raise ValueError("Unsupported file format. Please upload PDF, PNG, JPG, or JPEG.")

    if not images:
        raise ValueError("No valid image pages could be rendered from the uploaded file.")

    return images

def split_booklet_pages(images: List[Image.Image]) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Splits booklet images into:
    - first_page: Student & Exam Header page (Page 1)
    - remaining_pages: Answer sheets (Pages 2..N)
    """
    first_page = images[0]
    remaining_pages = images[1:] if len(images) > 1 else [first_page]
    return first_page, remaining_pages

def read_scanned_booklet(uploaded_file) -> List[Image.Image]:
    """
    Reads uploaded Streamlit UploadedFile or file object into PIL Images list.
    """
    if hasattr(uploaded_file, "getvalue") and hasattr(uploaded_file, "name"):
        return convert_bytes_to_images(uploaded_file.getvalue(), uploaded_file.name)
    elif isinstance(uploaded_file, (bytes, bytearray)):
        return convert_bytes_to_images(bytes(uploaded_file), "uploaded_file.pdf")
    return [uploaded_file] if isinstance(uploaded_file, Image.Image) else []

