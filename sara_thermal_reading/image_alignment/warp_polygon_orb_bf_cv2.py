from typing import cast

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

# This was the first version of the image alignment algorithm


def warp_polygon_orb_bf_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Warp the polygon from reference image to source image using ORB features and Homography.

    Args:
        reference_image: The image where the polygon is defined (Image1)
        source_image: The target image where we want to map the polygon (Image2)
        roi_polygon: Polygon points defined in reference image coordinates

    Returns:
        tuple: (warped_polygon_on_source, source_image)
    """
    # Define the polygon points in Image1 (x, y)
    polygon_points = np.array(roi_polygon, dtype=np.float32)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create()  # type: ignore
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        logger.error("Could not find features in one or both images")
        return polygon_points.reshape(-1, 1, 2), source_image

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    if len(matches) < 4:
        logger.error(f"Insufficient matches found: {len(matches)}. Need at least 4.")
        return polygon_points.reshape(-1, 1, 2), source_image

    # Calculate the homography matrix
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    if H is None:
        logger.error("Could not compute homography")
        return polygon_points.reshape(-1, 1, 2), source_image

    # Warp the polygon points from Image1 to Image2
    warped_polygon = cv2.perspectiveTransform(polygon_points.reshape(-1, 1, 2), H)

    return cast(NDArray[np.float32], warped_polygon), source_image
