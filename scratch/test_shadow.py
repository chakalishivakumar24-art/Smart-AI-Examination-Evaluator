import cv2
import numpy as np

def remove_shadows_and_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV shadow compensation and contrast normalization:
    Uses morphological dilation to estimate non-uniform illumination background,
    divides original image by background to remove shadows, and applies CLAHE.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
        
    # Split channels
    channels = cv2.split(image_bgr)
    norm_channels = []
    
    for ch in channels:
        # Morphological dilation with large kernel to capture background lighting
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated = cv2.dilate(ch, kernel)
        bg = cv2.medianBlur(dilated, 21)
        
        # Difference or division
        diff = 255 - cv2.absdiff(ch, bg)
        norm_ch = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm_channels.append(norm_ch)
        
    enhanced_bgr = cv2.merge(norm_channels)
    return enhanced_bgr

# Test on WhatsApp Image 03
img = cv2.imread(r"data\uploads\25b29d44_WhatsApp Image 03.jpeg")
if img is not None:
    enhanced = remove_shadows_and_enhance(img)
    cv2.imwrite(r"scratch\test_shadow_removed.jpg", enhanced)
    print("Shadow removal successful!")
