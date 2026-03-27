"""
Image Quality Analyzer — Smart Routing for Super-Resolution

Lightweight analysis module that extracts quality metrics from an input
image to determine which pipeline stages are necessary:

  - blur_score:     0.0 (sharp) → 1.0 (very blurry)
  - noise_level:    0.0 (clean) → 1.0 (very noisy)
  - detail_score:   0.0 (flat)  → 1.0 (highly detailed)
  - face_detected:  bool

Runs in <100ms on CPU. Zero GPU usage.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger("creator_ai.quality_analyzer")


class QualityAnalyzer:
    """Analyze image quality to drive conditional routing."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.blur_threshold = cfg.get("blur_threshold", 0.7)
        self.detail_threshold = cfg.get("detail_threshold", 0.5)
        self.noise_threshold = cfg.get("noise_threshold", 0.6)

        # Face detection — reuse facexlib if available, else OpenCV cascade
        self._face_detector = None
        try:
            from facexlib.detection import init_detection_model
            import torch
            device = torch.device("cpu")  # Detection on CPU is fast enough
            self._face_detector = init_detection_model(
                "retinaface_resnet50", half=False, device=device
            )
            self._det_method = "retinaface"
        except Exception:
            # Fallback to OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)
            self._det_method = "haar"

    def analyze(self, image_rgb: np.ndarray) -> dict:
        """
        Analyze image quality.

        Args:
            image_rgb: RGB uint8 (H, W, 3)

        Returns:
            dict with blur_score, noise_level, detail_score, face_detected
        """
        # Work on a small version for speed
        h, w = image_rgb.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = image_rgb

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        blur = self._compute_blur(gray)
        noise = self._compute_noise(gray)
        detail = self._compute_detail(gray)
        face = self._detect_face(small)

        result = {
            "blur_score": round(blur, 3),
            "noise_level": round(noise, 3),
            "detail_score": round(detail, 3),
            "face_detected": face,
        }

        # Print a compact summary
        face_icon = "👤" if face else "—"
        logger.info(f"[QualityAnalyzer] blur={blur:.2f}  noise={noise:.2f}  "
              f"detail={detail:.2f}  face={face_icon}")

        return result

    # ── Metric Implementations ──────────────────────────────

    @staticmethod
    def _compute_blur(gray: np.ndarray) -> float:
        """
        Laplacian variance → blur score.
        Low variance = blurry image.
        Returns 0.0 (sharp) → 1.0 (very blurry).
        """
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Typical range: 0 (flat) to ~2000+ (crisp)
        # Map: <50 → very blurry (1.0), >500 → sharp (0.0)
        score = 1.0 - min(lap_var / 500.0, 1.0)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _compute_noise(gray: np.ndarray) -> float:
        """
        High-frequency energy via Gaussian difference.
        Returns 0.0 (clean) → 1.0 (very noisy).
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred).astype(np.float32)
        noise_energy = diff.std()
        # Typical range: 2–20 for clean, 20–50 for noisy
        score = min(noise_energy / 30.0, 1.0)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _compute_detail(gray: np.ndarray) -> float:
        """
        Edge density via Canny.
        Returns 0.0 (flat/smooth) → 1.0 (highly detailed).
        """
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        # Typical: 0.01 (smooth) to 0.15+ (very detailed)
        score = min(edge_density / 0.10, 1.0)
        return max(0.0, min(score, 1.0))

    def _detect_face(self, image_rgb: np.ndarray) -> bool:
        """Detect if any face is present in the image."""
        try:
            if self._det_method == "retinaface":
                import torch
                img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                with torch.no_grad():
                    bboxes = self._face_detector.detect_faces(img_bgr, 0.5)
                return len(bboxes) > 0
            else:
                # OpenCV Haar cascade fallback
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                faces = self._face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                return len(faces) > 0
        except Exception:
            return False

    def get_routing(self, metrics: dict, mode: str = "fast") -> dict:
        """
        Determine which pipeline modules to activate.

        Args:
            metrics: Output from analyze()
            mode: "fast", "balanced", or "hd"

        Returns:
            dict of booleans for each optional stage
        """
        routing = {
            "denoise": metrics["noise_level"] > self.noise_threshold,
            "face_restore": metrics["face_detected"],
            "swinir": False,
            "diffusion": False,
        }

        if mode == "balanced":
            # SwinIR only if the image lacks detail
            routing["swinir"] = metrics["detail_score"] < self.detail_threshold

        elif mode == "hd":
            # SwinIR if detail is low, Diffusion if very blurry
            routing["swinir"] = metrics["detail_score"] < self.detail_threshold
            routing["diffusion"] = metrics["blur_score"] > self.blur_threshold

        return routing
