import base64
import io
import json
import logging
import statistics
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
from examples.load_train_data import load_train_data, get_by_id

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_BASE_URL = "http://192.168.1.100:1234/v1"
MODELS = [
    "qwen/qwen3-vl-8b",
    "qwen/qwen3-vl-8b:2",
    "google/gemma-3-12b",
    "google/gemma-3-12b:2",
]
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120
DEFAULT_CROP_MARGIN = 0.10
OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "ocr_results.csv"
TRAIN_DATA = load_train_data(
    "data/色差仪/爱色丽MA5QC色差仪/上汽江宁工厂-爱色丽MA-5-QC.xlsx"
)


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
def check_confidence(
    image: Image.Image,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = 256,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[float]:
    """
    Evaluate the confidence score for data extraction from an image.

    Args:
        image: PIL Image to evaluate.
        model: Model name to use for confidence evaluation.
        base_url: Base URL for the OpenAI-compatible API.
        max_tokens: Maximum tokens in response (default: 256 for score).
        timeout: Request timeout in seconds.

    Returns:
        Confidence score as float between 0 and 1, or None if evaluation fails.
    """
    try:
        image_base64 = encode_image_to_base64(image)
    except Exception as e:
        logger.error(f"Failed to encode image to base64 for confidence check: {e}")
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT["score"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    try:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        result = response.json()
        result_text = result["choices"][0]["message"]["content"].strip()

        # Parse confidence score from response
        confidence = float(result_text)

        # Validate score is in range [0, 1]
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Confidence score {confidence} out of range [0, 1]")
            return None

        return confidence

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during confidence check: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected API response format during confidence check: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to parse confidence score as float: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during confidence check: {e}")
        return None


@with_logging
def ocr_image(
    image: Image.Image,
    prompt: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
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
        "temperature": 0.0,
        "top_k": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
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
    required_fields = ["id", "deg"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field '{field}' in OCR result")
            return {}

    result = {
        "id": data.get("id", ""),
    }

    degrees = data.get("deg", [])
    if not degrees:
        logger.warning("No degree values found in OCR result")
        return result

    # Process measurement values (L, a, b, etc.)
    measurement_keys = [key for key in data.keys() if key not in ["id", "deg"]]
    numbers = []
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
            numbers.append(abs(value))
            result[f"{degree}°{key}*"] = value

    max_number = max(numbers)
    if max_number < 10:
        result["type"] = "relative"
    else:
        result["type"] = "absolute"
    return result


def validate_ocr_result(ocr_result: dict) -> tuple[str, bool]:
    """
    Validate OCR results against training data ground truth.

    Args:
        ocr_result: Dictionary containing OCR extracted data with flattened structure
                   (e.g., {"id": "...", "type": "...", "15°L*": ..., "25°L*": ...})

    Returns:
        Tuple of (comments_string, has_mismatch):
            - comments_string: Comma-separated list of mismatches, empty if no issues
            - has_mismatch: Boolean indicating if validation failed
    """
    # Validate input has required fields
    if not ocr_result:
        logger.warning("Empty OCR result provided for validation")
        return "Empty OCR result", True

    if "id" not in ocr_result:
        logger.warning("OCR result missing 'id' field")
        return "Missing 'id' field in OCR result", True

    # Get training data for comparison
    try:
        train_data = get_by_id(TRAIN_DATA, ocr_result["id"])
    except Exception as e:
        logger.error(
            f"Error retrieving training data for id '{ocr_result.get('id')}': {e}"
        )
        return f"Failed to retrieve training data: {e}", True

    if train_data is None:
        logger.warning(f"No training data found for id: {ocr_result['id']}")
        return f"No training data found for id: {ocr_result['id']}", True

    # Extract measurement keys (exclude metadata fields)
    measurement_keys = [
        key
        for key in ocr_result.keys()
        if key not in ["id", "file_name", "success", "comments", "has_mismatch"]
    ]

    if not measurement_keys:
        logger.warning("No measurement keys found in OCR result")
        return "No measurement data to validate", True

    # Compare each measurement with training data
    comments = []
    for key in measurement_keys:
        ocr_value = ocr_result[key]

        # Convert key to lowercase for train_data lookup (e.g., "15°L*" -> "15°l*")
        train_key = key.lower()

        # Check if key exists in training data
        if train_key not in train_data:
            logger.warning(f"Key '{train_key}' not found in training data")
            comments.append(f"{key}: Not found in training data")
            continue

        target_value = train_data[train_key]

        # Handle None/NaN values
        if pd.isna(ocr_value) and pd.isna(target_value):
            continue  # Both are NaN, consider them equal

        if pd.isna(ocr_value):
            comments.append(f"{key}: OCR returned NULL, expected {target_value}")
            continue

        if pd.isna(target_value):
            comments.append(f"{key}: Expected NULL, got {ocr_value}")
            continue

        # Compare values as strings (normalized)
        ocr_str = str(ocr_value).strip()
        target_str = str(target_value).strip()

        if ocr_str != target_str:
            comments.append(f"{key}: OCR={ocr_str} ≠ Target={target_str}")

    comments_str = ", ".join(comments)
    has_mismatch = len(comments) > 0

    if has_mismatch:
        logger.warning(f"Validation failed for id '{ocr_result['id']}': {comments_str}")
    else:
        logger.info(f"Validation passed for id '{ocr_result['id']}'")

    return comments_str, has_mismatch


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
    model: str,
    crop_margin: float = DEFAULT_CROP_MARGIN,
    confidence_threshold: float = 0.3,
) -> Optional[dict]:
    """
    Process a detected image: crop, save, check confidence, and perform OCR.

    Args:
        pipeline: GroundingDINOBase instance for image processing.
        image: Original PIL Image.
        file_identifier: Unique identifier for the file.
        device_prompt: Prompt for OCR specific to the device type.
        model: Model name to use for OCR.
        crop_margin: Margin for cropping (default: 0.10).
        confidence_threshold: Minimum confidence score to proceed with OCR (default: 0.3).

    Returns:
        Dictionary containing OCR results and confidence score, or None if processing fails.
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
        # Continue with confidence check even if save fails

    # Check confidence before OCR
    logger.info(f"Checking confidence for: {file_identifier} with model: {model}")
    confidence = check_confidence(cropped, model=model)

    if confidence is None:
        logger.warning(f"Failed to get confidence score for {file_identifier}")
        # Continue with OCR anyway if confidence check fails
    elif confidence < confidence_threshold:
        logger.warning(
            f"Low confidence score ({confidence:.2f}) for {file_identifier}, "
            f"below threshold {confidence_threshold}. Proceeding with OCR anyway."
        )
    else:
        logger.info(f"Confidence score: {confidence:.2f} for {file_identifier}")

    # Perform OCR
    logger.info(f"Running OCR on cropped image: {file_identifier} with model: {model}")
    ocr_result = ocr_image(cropped, prompt=device_prompt, model=model)

    if not ocr_result:
        logger.warning("OCR returned no results")
        return None

    # Flatten OCR result
    flattened_result = flatten_ocr_result(ocr_result)
    if not flattened_result:
        logger.warning("Failed to flatten OCR result")
        return None

    # Add confidence score to result
    flattened_result["confidence"] = confidence

    return flattened_result


def handle_image(
    pipeline: GroundingDINOBase,
    file: Path,
    image: Image.Image,
    model: str,
) -> None:
    """
    Process a single image: detect, crop, OCR, and save results.

    Args:
        pipeline: GroundingDINOBase instance for image processing.
        file: Path to the image file.
        image: PIL Image object.
        model: Model name to use for OCR (default: DEFAULT_MODEL).
    """
    logger.info(f"Processing: {file.name} with model: {model}")

    file_identifier = generate_file_identifier(file)

    result_data = {"file_name": file_identifier, "success": False, "model": model}

    try:
        # Detect objects in the image
        detection = pipeline.process_image(image)
        if not detection:
            logger.info(f"No objects detected in {file.name}")
            append_to_csv(result_data, OUTPUT_CSV)
            return

        # Process detected image (crop + confidence check + OCR)
        ocr_data = process_detected_image(
            pipeline=pipeline,
            image=image,
            file_identifier=file_identifier,
            device_prompt=PROMPT["爱色丽MA5QC色差仪"],
            model=model,
        )

        if not ocr_data:
            logger.warning(f"Failed to process detected image: {file.name}")
            append_to_csv(result_data, OUTPUT_CSV)
            return

        # Update result with OCR data
        result_data.update(ocr_data)
        result_data["success"] = True

        # Validate OCR results against training data
        comments, has_mismatch = validate_ocr_result(ocr_data)
        result_data["comments"] = comments
        result_data["has_mismatch"] = has_mismatch

        append_to_csv(result_data, OUTPUT_CSV)

        if has_mismatch:
            logger.warning(f"Processed {file.name} with validation mismatches")
        else:
            logger.info(f"Successfully processed: {file.name}")

    except Exception as e:
        logger.error(f"Error processing image {file.name}: {e}", exc_info=True)
        append_to_csv(result_data, OUTPUT_CSV)


def main(model: str) -> None:
    """
    Process all images in the 'images' directory with specified model.

    Args:
        model: Model name to use for OCR (default: DEFAULT_MODEL).
    """

    pipeline = GroundingDINOBase()
    logger.info(f"Starting processing with model: {model}")

    for file, image in image_loader("data/色差仪/爱色丽MA5QC色差仪"):
        handle_image(pipeline, file, image, model=model)

    logger.info(f"Completed processing with model: {model}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting batch processing with {len(MODELS)} models: {MODELS}")

    for model in MODELS:
        logger.info(f"=" * 80)
        logger.info(f"Processing with model: {model}")
        logger.info(f"=" * 80)
        main(model)

    logger.info("Batch processing completed for all models")
