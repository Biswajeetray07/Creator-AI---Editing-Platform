"""
Text Detector — EasyOCR-based (bundles CRAFT internally).

EasyOCR uses CRAFT for text detection and a recognition head for reading.
We only need the detection stage (bounding boxes), not recognition.

Fallback: If EasyOCR fails, we use OpenCV morphological text detection.
"""
import numpy as np
import cv2

try:
    import easyocr
except ImportError:
    easyocr = None


class TextDetector:
    """
    Detects text regions using EasyOCR (CRAFT-based) with morphological fallback.
    Returns polygon coordinates for each detected text region.
    """

    def __init__(self, device: str = "cuda", languages: list = None, **kwargs):
        self.reader = None
        self.languages = languages or ["en"]

        if easyocr is None:
            print("[TextDetector] easyocr not installed. Using morphological fallback.")
            return

        gpu = device == "cuda"
        try:
            self.reader = easyocr.Reader(
                self.languages, gpu=gpu, verbose=False
            )
            print(f"[TextDetector] EasyOCR (CRAFT) initialized (GPU: {gpu}).")
        except Exception as e:
            print(f"[TextDetector] EasyOCR init failed: {e}. Using fallback.")

    def __call__(self, image: np.ndarray) -> list:
        """
        Detect text in an image.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            List of polygons. Each polygon is an np.ndarray of shape (N, 2).
            For rectangular detections N=4.
        """
        polygons = []

        # --- Primary: EasyOCR (CRAFT-based) ---
        if self.reader is not None:
            try:
                # EasyOCR returns: [ ([x1,y1],[x2,y2],[x3,y3],[x4,y4]), text, conf ]
                # We convert RGB→BGR since easyocr expects BGR or file path
                results = self.reader.readtext(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    detail=1,
                    paragraph=False,
                    text_threshold=0.5,  # Lower = catches faint text
                    low_text=0.3,        # Lower = catches faint strokes
                    mag_ratio=1.5        # Slight magnification
                )
                for (bbox, text, conf) in results:
                    poly = np.array(bbox, dtype=np.int32)
                    polygons.append(poly)

                if len(polygons) > 0:
                    print(f"      [CRAFT/EasyOCR] Found {len(polygons)} text regions.")
                    return polygons
            except Exception as e:
                print(f"      [CRAFT/EasyOCR] Detection failed: {e}. Trying fallback...")

        # --- Fallback: Morphological text detection ---
        polygons = self._morphological_detect(image)
        print(f"      [Morphological Fallback] Found {len(polygons)} text regions.")
        return polygons

    def _morphological_detect(self, image: np.ndarray) -> list:
        """
        Simple morphological text detection using edge density.
        Works for large, high-contrast text on walls/signs.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold to find dark text on light backgrounds
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        # Connect nearby characters into text blocks
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_v)
        connected = cv2.dilate(connected, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = image.shape[:2]
        min_area = (h * w) * 0.001  # At least 0.1% of image
        max_area = (h * w) * 0.5    # At most 50% of image

        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(np.int32)
                polygons.append(box)

        return polygons
