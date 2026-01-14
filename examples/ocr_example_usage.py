"""
Example usage of the improved OCR function with OpenAI client and Pydantic.
This demonstrates how to use structured outputs for OCR tasks.
"""

import logging
from pathlib import Path

from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Import the improved OCR function
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from examples.process_and_ocr_img import ocr_image, OCRResponse
from prompt import PROMPT


def main():
    """Demonstrate OCR with structured outputs."""

    # Load a sample image
    image_path = Path("data/色差仪/爱色丽MA5QC色差仪").glob("*.jpg").__next__()
    image = Image.open(image_path)

    logger.info(f"Processing image: {image_path.name}")

    # Perform OCR with structured outputs
    result = ocr_image(
        image=image,
        prompt=PROMPT["爱色丽MA5QC色差仪"],
        model="qwen/qwen3-vl-8b",
    )

    if result:
        logger.info("OCR successful!")
        logger.info(f"Extracted data: {result}")

        # Validate with Pydantic model
        validated = OCRResponse.model_validate(result)
        logger.info(f"Validated response: {validated}")

        # Access typed fields
        logger.info(f"Sample ID: {validated.id}")
        logger.info(f"Degrees: {validated.deg}")
        logger.info(f"L values: {validated.L}")
        logger.info(f"a values: {validated.a}")
        logger.info(f"b values: {validated.b}")
    else:
        logger.error("OCR failed")


if __name__ == "__main__":
    main()
