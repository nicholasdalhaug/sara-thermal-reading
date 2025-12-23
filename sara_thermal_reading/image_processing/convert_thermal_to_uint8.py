import cv2
import numpy as np
from numpy.typing import NDArray


def convert_thermal_to_uint8(
    thermal_image: NDArray[np.float64],
    clip_range: tuple[float, float] | None = None,
) -> NDArray[np.uint8]:
    """
    Converts a thermal image (float64) to a uint8 image (0-255) using a specific clip range.

    Args:
        thermal_image: The input thermal image with temperature values (float64).
        clip_range: A tuple of (min_value, max_value) to use for clipping and normalization.
                    If None, uses the min and max values of the image.

    Returns:
        The converted uint8 image normalized between 0 and 255.
    """
    if clip_range is None:
        vmin = float(np.min(thermal_image))
        vmax = float(np.max(thermal_image))
    else:
        vmin, vmax = clip_range

    clipped_image = np.clip(thermal_image, vmin, vmax)

    normalized_image = np.zeros_like(clipped_image, dtype=np.uint8)
    cv2.normalize(
        clipped_image, normalized_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return normalized_image


def calculate_clip_values_from_percentiles(
    thermal_image: NDArray[np.float64],
    clip_percentile_min: float = 0.01,
    clip_percentile_max: float = 99.99,
) -> tuple[float, float]:
    """
    Calculates the min and max values for clipping based on percentiles.
    Useful to avoid a small amount of extreme values affecting normalization.
    For example if one pixel has an abnormally high or low temperature reading.

    NOTE: Do not calculate max or min temperature on the same image
    after applying these clip values as they will be clipped

    Args:
        thermal_image: The input thermal image with temperature values (float64).
        clip_percentile_min: The lower percentile to clip (default 0.01%).
        clip_percentile_max: The upper percentile to clip (default 99.99%).

    Returns:
        A tuple containing (min_clip_value, max_clip_value).
    """
    vmin, vmax = np.percentile(
        thermal_image, [clip_percentile_min, clip_percentile_max]
    )
    return float(vmin), float(vmax)
