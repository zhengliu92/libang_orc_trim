import base64
import io
import logging
import sys
from pathlib import Path

import httpx
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dino_base import GroundingDINOBase, image_loader

logger = logging.getLogger(__name__)


def ocr_image(
    image: Image.Image,
    base_url: str = "http://192.168.1.100:1234/v1",
    prompt: str = "Please extract all text from this image. Return only the extracted text without any additional explanation.",
) -> str:
    """
    Perform OCR on an image using an OpenAI-compatible vision API.

    Args:
        image: PIL Image to perform OCR on.
        base_url: Base URL for the OpenAI-compatible API.
        prompt: Prompt for the OCR API.
    Returns:
        Extracted text from the image.
    """
    # Convert PIL Image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": "vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }

    # Use httpx with trust_env=False to ignore system proxy settings
    with httpx.Client(trust_env=False, timeout=60) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def main() -> None:
    """Process all images in the 'images' directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pipeline = GroundingDINOBase()

    for file, image in image_loader("images"):
        logger.info(f"Processing: {file.name}")

        detection = pipeline.process_image(image)

        if detection:
            cropped = pipeline.crop_image(image)
            if cropped:
                logger.info(f"Running OCR on cropped image: {file.name}")
                text = ocr_image(cropped)
                logger.info(f"OCR result for {file.name}:\n{text}")


if __name__ == "__main__":
    main()
