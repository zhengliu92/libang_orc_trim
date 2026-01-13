from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from modelscope import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image

import logging

from utils import with_logging

logger = logging.getLogger(__name__)


class DetectionResult:
    """Data class for storing detection results."""

    __slots__ = ("box", "score", "label", "box_area_ratio")

    def __init__(
        self,
        box: np.ndarray,
        score: float,
        label: str,
        box_area_ratio: float,
    ):
        self.box = box
        self.score = score
        self.label = label
        self.box_area_ratio = box_area_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box,
            "score": self.score,
            "label": self.label,
            "box_area_ratio": self.box_area_ratio,
        }


class GroundingDINOBase:
    """Grounding DINO model wrapper for zero-shot object detection and cropping."""

    def __init__(
        self,
        model_path: str = "model/GroundingDINOBase",
        default_prompt: str = "screen",
        output_dir: str = "output",
        box_threshold: float = 0.1,
        text_threshold: float = 0.3,
        min_box_area_ratio: float = 0.05,
        device: Optional[str] = None,
    ):
        """
        Initialize the GroundingDINO model pipeline.

        Args:
            model_path: Path to the pretrained model.
            default_prompt: Default text prompt for detection.
            output_dir: Directory for saving output images.
            box_threshold: Confidence threshold for box detection.
            text_threshold: Confidence threshold for text matching.
            min_box_area_ratio: Minimum box area ratio to filter small detections.
            device: Device to run inference on ('cuda' or 'cpu').
        """
        self.model_path = model_path
        self.default_prompt = default_prompt
        self.output_dir = Path(output_dir)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_box_area_ratio = min_box_area_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[AutoModelForZeroShotObjectDetection] = None
        self._detection: Optional[DetectionResult] = None

        self._init_output_dirs()

    def _init_output_dirs(self) -> None:
        """Create output directories for cropped and detection images."""
        (self.output_dir / "crop").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "detection").mkdir(parents=True, exist_ok=True)

    @property
    def detection(self) -> Optional[DetectionResult]:
        """Get the current detection result."""
        return self._detection

    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._processor is not None and self._model is not None

    def load_model(self) -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
        """
        Lazy load the model and processor.

        Returns:
            Tuple of (processor, model).
        """
        if self.is_model_loaded:
            return self._processor, self._model

        logger.info(f"Loading model from {self.model_path} on {self.device}")
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path
        ).to(self.device)
        logger.info("Model loaded successfully")

        return self._processor, self._model

    def _normalize_prompt(self, text_prompt: Optional[str] = None) -> str:
        """Normalize prompt text to lowercase with trailing period."""
        prompt = (text_prompt or self.default_prompt).strip().lower()
        if prompt and not prompt.endswith("."):
            prompt += "."
        return prompt

    def _filter_detections(
        self,
        detections: list[dict[str, Any]],
        image_pixel_size: int,
    ) -> Optional[DetectionResult]:
        """
        Filter detections by area ratio and select the best one.

        Args:
            detections: Raw detection results from the model.
            image_pixel_size: Total pixel count of the image.

        Returns:
            Best detection result or None if no valid detections.
        """
        self._detection = None
        rows: list[dict[str, Any]] = []

        for result in detections:
            boxes = result["boxes"].cpu().numpy().astype(int)
            scores = result["scores"].cpu().numpy()
            labels = result["text_labels"]

            for box, score, label in zip(boxes, scores, labels):
                if not label:
                    continue

                x0, y0, x1, y1 = map(int, box)
                box_area = (x1 - x0) * (y1 - y0)
                box_area_ratio = round(box_area / image_pixel_size, 3)

                if box_area_ratio < self.min_box_area_ratio:
                    continue

                rows.append(
                    {
                        "box": box,
                        "score": score,
                        "label": label,
                        "box_area_ratio": box_area_ratio,
                    }
                )

        if not rows:
            logger.debug("No valid detections found after filtering")
            return None

        # Select best detection by score
        df = pd.DataFrame(rows)
        if df.shape[0] > 1:
            df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        best = df.iloc[0]
        self._detection = DetectionResult(
            box=best["box"],
            score=best["score"],
            label=best["label"],
            box_area_ratio=best["box_area_ratio"],
        )

        return self._detection

    @with_logging
    def process_image(
        self,
        image: Image.Image,
        text_prompt: Optional[str] = None,
    ) -> Optional[DetectionResult]:
        """
        Run object detection on an image.

        Args:
            image: PIL Image to process.
            text_prompt: Optional text prompt for detection.

        Returns:
            DetectionResult if found, None otherwise.
        """
        processor, model = self.load_model()
        prompt = self._normalize_prompt(text_prompt)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [image.size[::-1]]
        detections = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        image_size = image.width * image.height
        return self._filter_detections(detections, image_size)

    @with_logging
    def crop_image(
        self, image: Image.Image, padding: float = None
    ) -> Optional[Image.Image]:
        """
        Crop the detected region from the image.

        Args:
            image: PIL Image to crop.

        Returns:
            Cropped PIL Image or None if no detection available.
        """
        if not self._detection:
            logger.warning("No valid detection available for cropping")
            return None

        x0, y0, x1, y1 = map(int, self._detection.box)

        # Clamp coordinates to image bounds
        x0 = max(0, min(x0, image.width - 1))
        y0 = max(0, min(y0, image.height - 1))
        x1 = max(x0 + 1, min(x1, image.width))
        y1 = max(y0 + 1, min(y1, image.height))

        crop_image_size = ((x1 - x0), (y1 - y0))
        if padding:
            padding_x = crop_image_size[0] * padding
            padding_y = crop_image_size[1] * padding
            x0 -= padding_x
            x1 += padding_x
            y0 -= padding_y
            y1 += padding_y
        return image.crop((x0, y0, x1, y1))

    def save_image(self, image: Image.Image, file_name: str) -> str:
        """
        Save cropped image to disk.

        Args:
            image: PIL Image to save.
            file_name: Base name for the output file (without extension).

        Returns:
            Path to the saved file.
        """
        output_file = self.output_dir / "crop" / f"{file_name}.png"
        image.save(output_file, format="PNG")
        return str(output_file)

    def _encode_and_save(self, image_bgr: np.ndarray, output_path: Path) -> str:
        """Encode and save image to disk."""
        success, encoded_image = cv2.imencode(".png", image_bgr)
        if not success:
            raise RuntimeError(f"Failed to encode image for saving: {output_path}")
        encoded_image.tofile(str(output_path))
        return str(output_path)

    def _draw_failed_detection(self, image_bgr: np.ndarray) -> np.ndarray:
        """Draw 'No valid detection' text on image."""
        cv2.putText(
            image_bgr,
            "No valid detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return image_bgr

    def _draw_detection_box(
        self,
        image_bgr: np.ndarray,
        detection: DetectionResult,
    ) -> np.ndarray:
        """Draw detection box and label on image."""
        x0, y0, x1, y1 = map(int, detection.box)
        text = f"{detection.label}: {detection.score:.2f}"

        # Draw bounding box
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Calculate text position
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = max(y0 - 5, text_height + 5)

        # Draw text background
        bg_top_left = (x0, text_y - text_height - baseline)
        bg_bottom_right = (x0 + text_width, text_y + baseline)
        cv2.rectangle(image_bgr, bg_top_left, bg_bottom_right, (0, 0, 255), -1)

        # Draw text
        cv2.putText(
            image_bgr,
            text,
            (x0, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return image_bgr

    def save_detection_image(
        self,
        image: Image.Image,
        save_file_name: str,
    ) -> str:
        """
        Save image with detection visualization.

        Args:
            image: Original PIL Image.
            save_file_name: Base name for the output file.

        Returns:
            Path to the saved detection image.
        """
        base_output = self.output_dir / "detection" / f"{save_file_name}_detection.png"
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if not self._detection:
            failed_output = base_output.with_name(
                f"{base_output.stem}_failed{base_output.suffix}"
            )
            self._draw_failed_detection(image_bgr)
            return self._encode_and_save(image_bgr, failed_output)

        self._draw_detection_box(image_bgr, self._detection)
        return self._encode_and_save(image_bgr, base_output)


def load_local_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from a local file path.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB format.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    supported_formats = {".png", ".jpg", ".jpeg"}
    if path.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported format: {path.suffix}. Supported: {supported_formats}"
        )

    return Image.open(path).convert("RGB")


def image_loader(
    image_dir: str | Path,
) -> Generator[tuple[Path, Image.Image], None, None]:
    """
    Load images from a directory.

    Args:
        image_dir: Path to the directory containing images.

    Yields:
        Tuple of (file_path, PIL Image) for each image.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory or contains no images.
    """
    path = Path(image_dir)

    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Provided path is not a directory: {path}")

    extensions = ("*.png", "*.jpg", "*.jpeg")
    image_files = [f for ext in extensions for f in path.rglob(ext)]

    if not image_files:
        raise ValueError(f"No images found in {path}")

    for file in sorted(image_files):
        with Image.open(file) as img:
            yield file, img.convert("RGB")
