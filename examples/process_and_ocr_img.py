import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dino_base import GroundingDINOBase, image_loader
from prompt import PROMPT
from utils import with_logging

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BASE_URL = "http://192.168.1.100:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-vl-8b:2"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 60
DEFAULT_CROP_MARGIN = 0.10
OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "ocr_results.csv"


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image to encode.

    Returns:
        Base64 encoded image string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_ocr_response(response_text: str) -> dict:
    """
    Parse OCR response text that may contain JSON wrapped in markdown code blocks.

    Args:
        response_text: Raw response text from OCR API.

    Returns:
        Parsed JSON dictionary.

    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON.
    """
    # Strip markdown code blocks if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    return json.loads(cleaned_text.strip())


@with_logging
def ocr_image(
    image: Image.Image,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """
    Perform OCR on an image using an OpenAI-compatible vision API.

    Args:
        image: PIL Image to perform OCR on.
        prompt: Prompt for the OCR API.
        base_url: Base URL for the OpenAI-compatible API.
        model: Model name to use for OCR.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.

    Returns:
        Extracted data as dictionary, or None if OCR fails.
    """
    try:
        image_base64 = encode_image_to_base64(image)
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    try:
        # Use httpx with trust_env=False to ignore system proxy settings
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        result = response.json()
        result_text = result["choices"][0]["message"]["content"]
        return parse_ocr_response(result_text)

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during OCR request: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected API response format: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OCR response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during OCR: {e}")
        return None


def flatten_ocr_result(data: dict) -> dict:
    """
    Flatten OCR result by expanding degree-based measurements into separate columns.

    Args:
        data: OCR result dictionary with structure:
            {
                "id": str,
                "type": str,
                "deg": List[int],
                "L": List[float],
                "a": List[float],
                "b": List[float]
            }

    Returns:
        Flattened dictionary with columns like "15°L*", "25°L*", etc.
    """
    if not data:
        return {}

    # Validate required fields
    required_fields = ["id", "type", "deg"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field '{field}' in OCR result")
            return {}

    result = {
        "id": data.get("id", ""),
        "type": data.get("type", ""),
    }

    degrees = data.get("deg", [])
    if not degrees:
        logger.warning("No degree values found in OCR result")
        return result

    # Process measurement values (L, a, b, etc.)
    measurement_keys = [key for key in data.keys() if key not in ["id", "type", "deg"]]

    for key in measurement_keys:
        values = data[key]
        if not isinstance(values, list):
            logger.warning(f"Unexpected non-list value for key '{key}'")
            continue

        if len(values) != len(degrees):
            logger.warning(
                f"Mismatch between degrees ({len(degrees)}) and values ({len(values)}) for key '{key}'"
            )
            continue

        for degree, value in zip(degrees, values):
            result[f"{degree}°{key}*"] = value

    return result


def generate_file_identifier(file: Path) -> str:
    """
    Generate a unique identifier for a file based on its parent path and name.

    Args:
        file: Path object for the file.

    Returns:
        Sanitized file identifier string.
    """
    parent_name = str(file.parent).replace("/", "_").replace("-", "_")
    return f"{parent_name}_{file.stem}"


def append_to_csv(data: dict, csv_path: Path) -> None:
    """
    Append a dictionary of data to a CSV file.

    Args:
        data: Dictionary containing the data to append.
        csv_path: Path to the CSV file.
    """
    if not data:
        logger.warning("No data to append to CSV")
        return

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([data])

        # Append to CSV, write header only if file doesn't exist
        df.to_csv(
            csv_path,
            mode="a",
            index=False,
            header=not csv_path.exists(),
            encoding="utf-8-sig",
        )
        logger.info(f"Appended results to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to append data to CSV: {e}")
        raise


def process_detected_image(
    pipeline: GroundingDINOBase,
    image: Image.Image,
    file_identifier: str,
    device_prompt: str,
    crop_margin: float = DEFAULT_CROP_MARGIN,
) -> Optional[dict]:
    """
    Process a detected image: crop, save, and perform OCR.

    Args:
        pipeline: GroundingDINOBase instance for image processing.
        image: Original PIL Image.
        file_identifier: Unique identifier for the file.
        device_prompt: Prompt for OCR specific to the device type.
        crop_margin: Margin for cropping (default: 0.10).

    Returns:
        Dictionary containing OCR results, or None if processing fails.
    """
    # Crop the image
    cropped = pipeline.crop_image(image, crop_margin)
    if not cropped:
        logger.warning("Failed to crop image")
        return None

    # Save cropped image
    try:
        pipeline.save_image(cropped, file_identifier)
    except Exception as e:
        logger.error(f"Failed to save cropped image: {e}")
        # Continue with OCR even if save fails

    # Perform OCR
    logger.info(f"Running OCR on cropped image: {file_identifier}")
    ocr_result = ocr_image(cropped, prompt=device_prompt)

    if not ocr_result:
        logger.warning("OCR returned no results")
        return None

    # Flatten OCR result
    flattened_result = flatten_ocr_result(ocr_result)
    if not flattened_result:
        logger.warning("Failed to flatten OCR result")
        return None

    return flattened_result


def handle_image(pipeline: GroundingDINOBase, file: Path, image: Image.Image) -> None:
    """
    Process a single image: detect, crop, OCR, and save results.

    Args:
        pipeline: GroundingDINOBase instance for image processing.
        file: Path to the image file.
        image: PIL Image object.
    """
    logger.info(f"Processing: {file.name}")

    file_identifier = generate_file_identifier(file)
    result_data = {"file_name": file_identifier, "success": False}

    try:
        # Detect objects in the image
        detection = pipeline.process_image(image)
        if not detection:
            logger.info(f"No objects detected in {file.name}")
            append_to_csv(result_data, OUTPUT_CSV)
            return

        # Process detected image (crop + OCR)
        ocr_data = process_detected_image(
            pipeline=pipeline,
            image=image,
            file_identifier=file_identifier,
            device_prompt=PROMPT["爱色丽MA5QC色差仪"],
        )

        if not ocr_data:
            logger.warning(f"Failed to process detected image: {file.name}")
            append_to_csv(result_data, OUTPUT_CSV)
            return

        # Update result with OCR data
        result_data.update(ocr_data)
        result_data["success"] = True
        append_to_csv(result_data, OUTPUT_CSV)
        logger.info(f"Successfully processed: {file.name}")

    except Exception as e:
        logger.error(f"Error processing image {file.name}: {e}", exc_info=True)
        append_to_csv(result_data, OUTPUT_CSV)


def main() -> None:
    """Process all images in the 'images' directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pipeline = GroundingDINOBase()

    for file, image in image_loader(
        "data/色差仪/爱色丽MA5QC色差仪/上汽江宁工厂-爱色丽MA-5-QC-相对值"
    ):
        handle_image(pipeline, file, image)


if __name__ == "__main__":
    main()
