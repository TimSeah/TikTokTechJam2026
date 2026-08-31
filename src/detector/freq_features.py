from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageOps


def radial_fft_features(
    image: Image.Image, bins: int = 32, image_size: int = 256
) -> np.ndarray:
    if bins < 1 or image_size < 2:
        raise ValueError("bins must be positive and image_size must be at least 2")

    grayscale = ImageOps.grayscale(image).resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    values = np.asarray(grayscale, dtype=np.float32) / 255.0
    values -= values.mean()
    window = np.outer(np.hanning(image_size), np.hanning(image_size)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(values * window))))

    coordinates = np.arange(image_size, dtype=np.float32) - (image_size - 1) / 2
    y_grid, x_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(x_grid**2 + y_grid**2)
    radius /= radius.max()

    edges = np.linspace(0.0, 1.0, bins + 1)
    features = np.empty(bins, dtype=np.float32)
    for index in range(bins):
        if index == bins - 1:
            mask = (radius >= edges[index]) & (radius <= edges[index + 1])
        else:
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
        features[index] = float(spectrum[mask].mean()) if mask.any() else 0.0
    return features


def batch_radial_fft_features(
    images: Iterable[Image.Image], bins: int = 32, image_size: int = 256
) -> np.ndarray:
    rows = [
        radial_fft_features(image, bins=bins, image_size=image_size) for image in images
    ]
    if not rows:
        return np.empty((0, bins), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)
