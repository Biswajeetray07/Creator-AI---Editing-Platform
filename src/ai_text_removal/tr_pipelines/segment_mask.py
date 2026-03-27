"""
Text Mask Generator — Polygon Fill.

For text regions, the detected text polygons from CRAFT/EasyOCR tightly wrap the text.
We simply fill these polygons to create the binary mask.
LaMa handles reconstructing the background within these regions.
"""
import cv2
import numpy as np


class TextMaskGenerator:
    """
    Converts detected text polygons into a precise stroke-level binary mask.
    Uses Otsu's thresholding to separate text from background, reducing the mask size
    drastically to preserve the original image. Applies dilation to cover anti-aliased
    edges and prevent "ghost" letters.
    """

    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, polygons: list) -> np.ndarray:
        """
        Generate a binary mask from text detection polygons.
        """
        h, w = image.shape[:2]
        global_mask = np.zeros((h, w), dtype=np.uint8)

        if len(polygons) == 0:
            return global_mask

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            if pts.ndim == 1:
                continue
                
            x, y, pw, ph = cv2.boundingRect(pts)
            x, y = max(0, x), max(0, y)
            pw = min(w - x, pw)
            ph = min(h - y, ph)
            
            if pw < 2 or ph < 2:
                continue

            roi = gray[y:y+ph, x:x+pw]
            
            try:
                # Otsu thresholding to find strokes
                _, t1 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, t2 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            except cv2.error:
                cv2.fillPoly(global_mask, [pts], 255)
                continue
                
            # Text usually occupies less area than background in a tight bounding box
            if np.count_nonzero(t1) < np.count_nonzero(t2):
                stroke_mask = t1
            else:
                stroke_mask = t2
                
            # CRITICAL: Dilate the strokes to cover the 2-3px anti-aliased halos!
            # The higher iterations (e.g. 4) ensure all faint ghost boundaries are swallowed.
            kernel = np.ones((5, 5), np.uint8)
            dilated_strokes = cv2.dilate(stroke_mask, kernel, iterations=4)
            
            # Constrain to the original polygon
            poly_shifted = pts - [x, y]
            poly_mask = np.zeros_like(roi)
            cv2.fillPoly(poly_mask, [poly_shifted], 255)
            
            final_roi_mask = cv2.bitwise_and(dilated_strokes, poly_mask)
            
            global_mask[y:y+ph, x:x+pw] = cv2.bitwise_or(global_mask[y:y+ph, x:x+pw], final_roi_mask)

        return global_mask
