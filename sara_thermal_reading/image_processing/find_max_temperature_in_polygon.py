from typing import cast

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray


def find_max_temperature_in_polygon(
    thermal_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    temp_range: tuple[float, float] = (20.0, 100.0),
) -> tuple[float, tuple[int, int]]:
    """
    Find the maximum temperature within a polygon region of a thermal image.

    Maps pixel values to actual temperature values based on the thermal image's color scale.

    Args:
        thermal_image: The thermal image array (BGR format from OpenCV)
        polygon_points: Array of polygon vertices in format [[x1, y1], [x2, y2], ...]
        temp_range: Tuple of (min_temp, max_temp) representing the temperature scale of the image

    Returns:
        Tuple of (max_temperature_celsius, (x_coord, y_coord)) where the max temperature was found
    """
    # Convert polygon points to integer coordinates
    polygon_pts = polygon_points.reshape(-1, 2).astype(np.int32)

    # Create a mask for the polygon region
    height, width = thermal_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], (255,))

    # For thermal images, we need to extract temperature information more intelligently
    # Method 1: Try to use the thermal colormap information
    if len(thermal_image.shape) == 3:
        # Convert BGR to HSV for better thermal analysis
        hsv_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2HSV)

        # For thermal images, often the Value (brightness) or Hue channel contains temperature info
        # Use the Value channel as it typically correlates with temperature intensity
        thermal_intensity = hsv_thermal[:, :, 2]  # Value channel

        # Also try the original grayscale conversion as backup
        gray_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

        # Use the channel that shows more variation in the masked region
        masked_hsv_v = cv2.bitwise_and(thermal_intensity, thermal_intensity, mask=mask)
        masked_gray = cv2.bitwise_and(gray_thermal, gray_thermal, mask=mask)

        # Calculate standard deviation to see which channel has more information
        hsv_std = (
            np.std(masked_hsv_v[masked_hsv_v > 0]) if np.any(masked_hsv_v > 0) else 0
        )
        gray_std = (
            np.std(masked_gray[masked_gray > 0]) if np.any(masked_gray > 0) else 0
        )

        if hsv_std > gray_std:
            temperature_channel = thermal_intensity
            logger.info("Using HSV Value channel for temperature analysis")
        else:
            temperature_channel = gray_thermal
            logger.info("Using grayscale conversion for temperature analysis")
    else:
        temperature_channel = thermal_image
        logger.info("Using grayscale image for temperature analysis")

    # Apply mask to get only the polygon region
    masked_thermal = cv2.bitwise_and(
        temperature_channel, temperature_channel, mask=mask
    )

    # Find the maximum and minimum values in the masked region for calibration
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_thermal, mask=mask)

    # Also get global min/max from the entire image for reference
    global_min, global_max, _, _ = cv2.minMaxLoc(temperature_channel)

    logger.info(f"Pixel value range in polygon: {min_val} - {max_val}")
    logger.info(f"Global pixel value range: {global_min} - {global_max}")
    logger.info(f"Maximum pixel value location: {max_loc}")

    # Map pixel values to actual temperature
    # Method: Linear mapping from pixel range to temperature range
    min_temp, max_temp = temp_range

    # Use the global range for mapping to maintain consistency with the thermal scale
    if global_max > global_min:
        # Linear interpolation: temp = min_temp + (pixel_val - global_min) * (max_temp - min_temp) / (global_max - global_min)
        actual_temperature = min_temp + (max_val - global_min) * (
            max_temp - min_temp
        ) / (global_max - global_min)
    else:
        # Fallback if no variation detected
        actual_temperature = (min_temp + max_temp) / 2
        logger.warning("No temperature variation detected, using average temperature")

    # Clamp temperature to reasonable bounds
    actual_temperature = max(min_temp, min(max_temp, actual_temperature))

    logger.info(
        f"Mapped temperature: {actual_temperature:.1f}°C (from pixel value {max_val})"
    )

    return float(actual_temperature), cast(tuple[int, int], max_loc)
