from __future__ import annotations

import random
import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB


def download_image_with_retry(
    image_url: str, max_retries: int = 3, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES
) -> Image.Image | None:
    if max_retries < 1:
        max_retries = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=15, stream=True)
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return None

            content = BytesIO()
            size = 0
            for chunk in response.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > max_bytes:
                    return None
                content.write(chunk)
            content.seek(0)

            return Image.open(content)
        except (requests.exceptions.RequestException, UnidentifiedImageError):
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt + random.uniform(0, 1))
    return None


def load_image(image: Image.Image | dict | str | None) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, dict) and "bytes" in image:
        try:
            img = Image.open(BytesIO(image["bytes"]))
        except (UnidentifiedImageError, OSError):
            img = None
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            img = download_image_with_retry(image_url=image, max_retries=3)
        else:
            try:
                img = Image.open(image)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                img = None
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return img.convert("RGB") if img is not None and img.mode != "RGB" else img
