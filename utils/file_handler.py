from utils.pdf_processor import save_uploaded_file, UPLOADS_DIR

def validate_and_save_upload(file_bytes: bytes, original_filename: str):
    """Validates file format and saves to data/uploads/ directory."""
    return save_uploaded_file(file_bytes, original_filename)
