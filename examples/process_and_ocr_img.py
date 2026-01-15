import base64
import io
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dino_base import GroundingDINOBase, image_loader
from prompt import PROMPT
from utils import with_logging
from examples.load_train_data import load_train_data, get_by_id
from structured_logprobs import add_logprobs

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class OCRResponse(BaseModel):
    """Pydantic model for OCR response structure."""

    id: Optional[str] = Field(
        None, description="Sample identifier (e.g., 'Sample 001' or 'MSZ 001 001')"
    )
    deg: Optional[list[int]] = Field(
        None, description="Degree measurements, typically [15, 25, 45, 75, 110]"
    )
    L: Optional[list[float]] = Field(
        None, description="L* color values (LAB color space, can be negative)"
    )
    a: Optional[list[float]] = Field(
        None, description="a* color values (LAB color space, can be negative)"
    )
    b: Optional[list[float]] = Field(
        None, description="b* color values (LAB color space, can be negative)"
    )


# Configuration constants
DEFAULT_BASE_URL = "http://192.168.1.100:8081/v1"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120
DEFAULT_CROP_MARGIN = 0.10
RELATIVE_ABSOLUTE_THRESHOLD = 10  # Values below 10 are considered relative measurements
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


@with_logging
def ocr_image(
    image: Image.Image,
    prompt: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[tuple[dict, dict]]:
    """
    Perform OCR on an image using an OpenAI-compatible vision API with structured outputs.

    Args:
        image: PIL Image to perform OCR on.
        prompt: Prompt for the OCR API.
        base_url: Base URL for the OpenAI-compatible API.
        model: Model name to use for OCR.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (ocr_data, logprobs) where:
            - ocr_data: Extracted data as dictionary
            - logprobs: Log probabilities as dictionary
        Returns None if OCR fails.
    """
    # Validate inputs early
    if not image:
        logger.error("Image parameter is None or empty")
        return None

    if not prompt or not prompt.strip():
        logger.error("Prompt parameter is empty")
        return None

    if not model or not model.strip():
        logger.error("Model parameter is empty")
        return None

    # Encode image to base64
    try:
        image_base64 = encode_image_to_base64(image)
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        return None

    # Initialize OpenAI client with custom httpx client
    # Create httpx client that ignores system proxy settings
    http_client = httpx.Client(
        trust_env=False,
        timeout=timeout,
    )

    try:
        client = OpenAI(
            base_url=base_url,
            api_key="dummy",  # Required but not used for local models
            http_client=http_client,
        )

        # Create chat completion with structured outputs

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=True,
            top_logprobs=1,
            response_format=OCRResponse,
        )

        # Add logprobs information
        completion_with_logprobs = add_logprobs(completion)

        # Validate logprobs early
        if not completion_with_logprobs:
            logger.error("Failed to add logprobs to completion")
            return None

        if not hasattr(completion_with_logprobs, "log_probs"):
            logger.error("No logprobs available in completion")
            return None

        # Parse response content
        content = completion.choices[0].message.content
        if not content:
            logger.error("Empty response content")
            return None

        # Parse JSON and validate with Pydantic
        ocr_data = json.loads(content)
        logprobs = completion_with_logprobs.log_probs[0]
        return ocr_data, logprobs

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OCR response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during OCR: {e}", exc_info=True)
        return None
    finally:
        # Ensure httpx client is closed
        http_client.close()


def flatten_ocr_result(
    data: dict, logprobs: Optional[dict] = None
) -> tuple[dict, dict]:
    """
    Flatten OCR result by expanding degree-based measurements into separate columns.

    Args:
        data: OCR result dictionary with structure:
            {
                "id": str,
                "deg": list[int],
                "L": list[float],
                "a": list[float],
                "b": list[float]
            }
        logprobs: Optional logprobs dictionary with structure mirroring data.
                  If provided, logprobs will be converted to probabilities and merged into result.

    Returns:
        Tuple of (flattened_result, flattened_probs):
            - flattened_result: Dictionary with columns like "15°L*", "25°L*", etc.
            - flattened_probs: Dictionary with probability columns like "15°L*_prob", "25°L*_prob", etc.
    """
    # Validate required fields early
    if not data:
        logger.warning("Empty data dictionary provided")
        return {}, {}

    required_fields = ["id", "deg"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field '{field}' in OCR result")
            return {}, {}

    result = {
        "id": data.get("id", ""),
    }

    degrees = data.get("deg", [])
    if not degrees:
        logger.warning("No degree values found in OCR result")
        return result, {}

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

    # Determine type based on values
    if numbers:
        max_number = max(numbers)
        if max_number < RELATIVE_ABSOLUTE_THRESHOLD:
            result["type"] = "relative"
        else:
            result["type"] = "absolute"

    # Flatten and merge logprobs if provided
    flattened_probs = {}
    if logprobs:
        flattened_probs = flatten_logprobs(logprobs, degrees)

    return result, flattened_probs


def flatten_logprobs(logprobs: dict, degrees: list[int]) -> dict:
    """
    Flatten logprobs by converting to probabilities and expanding into separate columns.

    Args:
        logprobs: Logprobs dictionary with structure mirroring OCR response:
            {
                "id": float (log probability),
                "deg": List[float] (log probabilities),
                "L": List[float] (log probabilities),
                "a": List[float] (log probabilities),
                "b": List[float] (log probabilities)
            }
        degrees: List of degree values (e.g., [15, 25, 45, 75, 110])

    Returns:
        Flattened dictionary with probability columns like "15°L*_prob", "25°L*_prob", etc.
        Log probabilities are converted to probabilities using math.exp().
    """
    if not logprobs or not degrees:
        return {}

    result = {}

    # Add probability for id field if available
    if "id" in logprobs and isinstance(logprobs["id"], (int, float)):
        result["id_prob"] = math.exp(logprobs["id"])

    # Add probability for deg field if available
    if "deg" in logprobs and isinstance(logprobs["deg"], list):
        deg_logprobs = logprobs["deg"]
        # Calculate average logprob for degrees, then convert to probability
        if deg_logprobs:
            try:
                avg_deg_logprob = sum(deg_logprobs) / len(deg_logprobs)
                result["deg_prob"] = math.exp(avg_deg_logprob)
            except (TypeError, ZeroDivisionError):
                logger.warning("Failed to calculate average probability for degrees")

    # Process measurement logprobs (L, a, b, etc.)
    measurement_keys = [key for key in logprobs.keys() if key not in ["id", "deg"]]

    for key in measurement_keys:
        logprob_values = logprobs[key]

        # Handle non-list values
        if not isinstance(logprob_values, list):
            logger.warning(f"Unexpected non-list logprob value for key '{key}'")
            continue

        # Check length match with degrees
        if len(logprob_values) != len(degrees):
            logger.warning(
                f"Mismatch between degrees ({len(degrees)}) and logprob values ({len(logprob_values)}) for key '{key}'"
            )
            continue

        # Create flattened columns like "15°L*_prob", "25°L*_prob"
        # Convert log probabilities to probabilities
        for degree, logprob_value in zip(degrees, logprob_values):
            result[f"{degree}°{key}*_prob"] = math.exp(logprob_value)

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
    measurement_keys = [key for key in ocr_result.keys() if key not in ["id"]]

    if not measurement_keys:
        logger.warning("No measurement keys found in OCR result")
        return "No measurement data to validate", True

    # Compare each measurement with training data
    comments = []
    for key in measurement_keys:
        if key == "type":
            continue
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
) -> Optional[tuple[dict, dict]]:
    """
    Process a detected image: crop, save, and perform OCR.

    Args:
        pipeline: GroundingDINOBase instance for image processing.
        image: Original PIL Image.
        file_identifier: Unique identifier for the file.
        device_prompt: Prompt for OCR specific to the device type.
        model: Model name to use for OCR.
        crop_margin: Margin for cropping (default: 0.10).

    Returns:
        Tuple of (flattened_result, flattened_probs) containing OCR results,
        or None if processing fails.
    """
    if not file_identifier or not file_identifier.strip():
        logger.error("File identifier is empty")
        return None

    if not device_prompt or not device_prompt.strip():
        logger.error("Device prompt is empty")
        return None

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

    # Perform OCR
    logger.info(f"Running OCR on cropped image: {file_identifier} with model: {model}")
    ocr_result_tuple = ocr_image(cropped, prompt=device_prompt, model=model)

    if not ocr_result_tuple:
        logger.warning("OCR returned no results")
        return None

    ocr_result, logprobs = ocr_result_tuple

    # Flatten OCR result
    flattened_result, flattened_logprobs = flatten_ocr_result(ocr_result, logprobs)

    if not flattened_result:
        logger.warning("Failed to flatten OCR result")
        return None

    return flattened_result, flattened_logprobs


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
        model: Model name to use for OCR.
    """
    # Validate inputs early
    if not pipeline:
        logger.error("Pipeline parameter is None")
        return

    if not file:
        logger.error("File parameter is None")
        return

    if not image:
        logger.error("Image parameter is None")
        return

    if not model or not model.strip():
        logger.error("Model parameter is empty")
        return

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

        # Process detected image (crop +  OCR)
        processed_result = process_detected_image(
            pipeline=pipeline,
            image=image,
            file_identifier=file_identifier,
            device_prompt=PROMPT["爱色丽MA5QC色差仪"],
            model=model,
        )

        if not processed_result:
            logger.warning(f"Failed to process detected image: {file.name}")
            append_to_csv(result_data, OUTPUT_CSV)
            return

        ocr_data, probs = processed_result

        # Update result with OCR data
        result_data.update(ocr_data)
        # Validate OCR results against training data
        comments, has_mismatch = validate_ocr_result(ocr_data)
        result_data.update(probs)
        result_data["success"] = True
        result_data["has_mismatch"] = has_mismatch
        result_data["comments"] = comments
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
        model: Model name to use for OCR.
    """
    # Validate inputs early
    if not model or not model.strip():
        logger.error("Model parameter is empty")
        return

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
    model = "gemma-3-12b-it-Q6_K"
    logger.info(f"Starting processing with model: {model}")
    main(model)
    logger.info("Processing completed")
