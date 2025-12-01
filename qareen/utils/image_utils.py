from __future__ import annotations

import logging
import random
import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def download_image_with_retry(
    image_url: str, max_retries: int = 3, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES
) -> Image.Image | None:
    if max_retries < 1:
        max_retries = 1

    for attempt in range(max_retries):
        try:
            with requests.get(image_url, timeout=10, stream=True) as response:
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            logger.warning(
                                "Image size %s exceeds limit %s", content_length, max_bytes
                            )
                            return None
                    except ValueError:
                        pass  # Invalid content-length header, continue downloading

                content = BytesIO()
                size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > max_bytes:
                        logger.warning("Image size exceeds limit %s during download", max_bytes)
                        return None
                    content.write(chunk)
                content.seek(0)

                with Image.open(content) as im:
                    im.load()
                    return im.copy()
        except (requests.exceptions.RequestException, UnidentifiedImageError, OSError) as e:
            logger.debug("Attempt %d failed to download image %s: %s", attempt + 1, image_url, e)
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt + random.uniform(0, 1))
    return None


def load_image(image: Image.Image | dict | str | None) -> Image.Image | None:
    if image is None:
        return None

    img: Image.Image | None = None
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, dict) and "bytes" in image:
        if not isinstance(image["bytes"], (bytes, bytearray)):
            logger.error("Invalid image bytes type: %s", type(image["bytes"]))
            return None
        try:
            with Image.open(BytesIO(image["bytes"])) as im:
                im.load()
                img = im.copy()
        except (UnidentifiedImageError, OSError, TypeError, ValueError) as e:
            logger.error("Failed to load image from dict bytes: %s", e)
            img = None
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            img = download_image_with_retry(image_url=image, max_retries=3)
        else:
            try:
                with Image.open(image) as im:
                    im.load()
                    img = im.copy()
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                logger.debug("Failed to load image from file %s: %s", image, e)
                img = None
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return img.convert("RGB") if img is not None and img.mode != "RGB" else img
