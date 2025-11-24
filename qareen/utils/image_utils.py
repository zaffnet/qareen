from __future__ import annotations

import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError


def download_image_with_retry(image_url: str, max_retries: int = 3) -> Image.Image | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except (requests.exceptions.RequestException, UnidentifiedImageError):
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)
    return None


def load_image(image: Image.Image | dict | str | None) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, dict) and "bytes" in image:
        img = Image.open(BytesIO(image["bytes"]))
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
