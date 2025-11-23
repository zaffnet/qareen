"""Image processing utilities."""

from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import TYPE_CHECKING

import requests
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def download_image_with_retry(
    image_url: str,
    max_retries: int = 3,
) -> Image.Image | None:
    """Download image from URL with retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)
    return None


def load_image(image: Image.Image | dict | str | None) -> Image.Image | None:
    """Load image from various input types."""
    if image is None:
        return None

    if isinstance(image, Image.Image):
        pass
    elif isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            image = download_image_with_retry(image_url=image, max_retries=3)
        else:
            try:
                image = Image.open(image)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                image = None
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if image is not None and image.mode != "RGB":
        image = image.convert("RGB")

    return image
