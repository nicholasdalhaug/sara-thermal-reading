import cv2
import numpy as np
from numpy.typing import NDArray


def find_temperature_in_polygon(
    thermal_image: NDArray[np.float64], polygon_points: NDArray[np.int32]
) -> float:
    mask = np.zeros(thermal_image.shape, dtype=np.int32)
    polygon_points = polygon_points.astype(np.int32)
    cv2.fillPoly(mask, [polygon_points], (1,))
    mask = mask.astype(bool)

    max_temp = np.max(thermal_image[mask]).astype(float)

    return max_temp
