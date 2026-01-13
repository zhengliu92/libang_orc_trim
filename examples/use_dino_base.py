import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dino_base import GroundingDINOBase, image_loader

logger = logging.getLogger(__name__)


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
                crop_path = pipeline.save_image(cropped, file.stem)
                logger.info(f"Cropped image saved: {crop_path}")

        detection_path = pipeline.save_detection_image(image, file.stem)
        logger.info(f"Detection image saved: {detection_path}")


if __name__ == "__main__":
    main()
