from __future__ import annotations

import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError


def download_image_with_retry(image_url: str, max_retries: int = 3) -> Image.Image | None:
    """
    Download an image from the given URL, retrying on transient network or decoding failures.
    
    Retries up to `max_retries` times with exponential backoff between attempts (2**attempt seconds). On success returns a PIL Image opened from the response content; if all attempts fail, returns `None`.
    
    Parameters:
        image_url (str): HTTP(S) URL of the image to download.
        max_retries (int): Maximum number of attempts to try downloading the image.
    
    Returns:
        Image.Image | None: A PIL Image if the download and decode succeed, `None` if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except (requests.exceptions.RequestException, UnidentifiedImageError):
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)
    # unreachable: loop always returns or raises
    return None  # type checker: satisfies return type annotation


def load_image(image: Image.Image | dict | str | None) -> Image.Image | None:
    """
    Normalize an input into a PIL Image in RGB mode or return None.
    
    Parameters:
        image (Image.Image | dict | str | None): Input to normalize. Accepted forms:
            - None: returns None.
            - PIL Image: returned (converted to RGB if not already).
            - dict with a "bytes" key: treated as raw image bytes (opened via BytesIO).
            - str starting with "http://" or "https://": treated as a URL and downloaded with retries.
            - str otherwise: treated as a filesystem path and opened.
    
    Returns:
        Image.Image | None: A PIL Image converted to RGB if possible, or None if the image could not be loaded.
    
    Raises:
        TypeError: If `image` is not one of the supported types.
    """
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