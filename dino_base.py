from pathlib import Path
import typing

import cv2
import numpy as np
import torch
from PIL import Image
from modelscope import AutoModelForZeroShotObjectDetection, AutoProcessor
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class GroundingDINOBase:
    def __init__(
        self,
        model_path: str = "model/GroundingDINOBase",
        default_prompt: str = "screen",
        output_dir: str = "output",
        box_threshold: float = 0.1,
        text_threshold: float = 0.3,
        min_box_area_ratio: int = 0.05,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.default_prompt = default_prompt
        self.output_dir = output_dir
        Path(self.output_dir, "crop").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "detection").mkdir(parents=True, exist_ok=True)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_box_area_ratio = min_box_area_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detection: dict | None = None

        self._processor: AutoProcessor | None = None
        self._model: AutoModelForZeroShotObjectDetection | None = None

    def load_model(self):
        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_path
            ).to(self.device)
        return self._processor, self._model

    def filter_detections(
        self,
        detections,
        image_pixel_size: int,
    ) -> typing.Dict[str, typing.Any] | None:
        self.detection = None
        rows: typing.List[typing.Dict[str, typing.Any]] = []
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
        if len(rows) == 0:
            return None
        ok_data = pd.DataFrame(rows)
        if ok_data.shape[0] > 1:
            ok_data = ok_data.sort_values(by=["score"], ascending=False).reset_index(
                drop=True
            )
            self.detection = ok_data.iloc[0].to_dict()
            return self.detection
        self.detection = ok_data.iloc[0].to_dict()
        return self.detection

    def process_image(
        self,
        image: Image.Image,
        text_prompt: str | None = None,
    ):

        processor, model = self.load_model()
        prompt = (text_prompt or self.default_prompt).strip().lower()
        if prompt and not prompt.endswith("."):
            prompt += "."
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
        self.filter_detections(detections, image_size)
        return

    def crop_save_image(
        self,
        image: Image.Image,
        save_file_name: str,
    ):
        if not self.detection:
            logger.warning(
                f"{save_file_name}: No valid detection available for cropping."
            )
            return
        output_file = Path(self.output_dir) / "crop" / f"{save_file_name}_crop.png"
        x0, y0, x1, y1 = map(int, self.detection["box"])
        x0 = max(0, min(x0, image.width))
        y0 = max(0, min(y0, image.height))
        x1 = max(x0 + 1, min(x1, image.width))
        y1 = max(y0 + 1, min(y1, image.height))
        cropped = image.crop((x0, y0, x1, y1))
        cropped.save(output_file, format="PNG")
        return str(output_file)

    def save_image(
        self,
        image: Image.Image,
        save_file_name: str,
    ):
        output_file = (
            Path(self.output_dir) / "detection" / f"{save_file_name}_detection.png"
        )
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not self.detection:
            failed_output = output_file.with_name(
                f"{output_file.stem}_failed{output_file.suffix}"
            )
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
            success, encoded_image = cv2.imencode(".png", image_bgr)
            if not success:
                raise RuntimeError("Failed to encode image for saving.")
            encoded_image.tofile(str(failed_output))
            return str(failed_output)

        box = self.detection["box"]
        score = self.detection["score"]
        label = self.detection["label"]
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = max(y0 - 5, text_height + 5)
        background_top_left = (x0, text_y - text_height - baseline)
        background_bottom_right = (x0 + text_width, text_y + baseline)
        cv2.rectangle(
            image_bgr,
            background_top_left,
            background_bottom_right,
            (0, 0, 255),
            thickness=-1,
        )
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
        success, encoded_image = cv2.imencode(".png", image_bgr)
        if not success:
            raise RuntimeError("Failed to encode image for saving.")
        encoded_image.tofile(str(output_file))
        return str(output_file)


def load_local_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Only PNG or JPG images are supported.")
    return Image.open(path).convert("RGB")


def image_loader(
    image_dir: str | Path,
) -> typing.Generator[typing.Tuple[Path, Image.Image], None, None]:
    path = Path(image_dir)
    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {path}")
    if not path.is_dir():
        raise ValueError("Provided path is not a directory.")
    suffixes = {".png", ".jpg", ".jpeg"}
    image_files: typing.List[Path] = sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in suffixes
    )
    if not image_files:
        raise ValueError(f"No PNG/JPG images found in {path}")
    for file in image_files:
        with Image.open(file) as img:
            yield (file, img.convert("RGB"))


def main():
    pipeline = GroundingDINOBase()
    for file, image in image_loader("images"):
        pipeline.process_image(image)
        pipeline.crop_save_image(image, file.stem)
        output_file = pipeline.save_image(image, file.stem)
        print(f"Processed image saved to: {output_file}")


if __name__ == "__main__":
    main()
