import argparse
import json
from typing import cast

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from numpy.typing import NDArray

from sara_thermal_reading.file_io.blob import BlobStorageLocation
from sara_thermal_reading.logger import setup_logger

setup_logger()
from loguru import logger

from sara_thermal_reading.file_io.file_utils import (
    REFERENCE_STORAGE_ACCOUNT,
    REFERENCE_STORAGE_CONNECTION_STRING,
    download_anonymized_image,
    load_reference_image_and_polygon,
    upload_to_visualized,
)


def check_reference_blob_exists(
    tag_id: str, inspection_description: str, installation_code: str
) -> bool:
    logger.info(
        f"Checking if reference blob exists for tag_id: {tag_id}, inspection_description: {inspection_description}, installation_code: {installation_code}"
    )

    ref_blob_service_client = BlobServiceClient.from_connection_string(
        REFERENCE_STORAGE_CONNECTION_STRING
    )
    img_path = f"{tag_id}_{inspection_description}/reference_image.jpeg"
    blob_client = ref_blob_service_client.get_blob_client(
        container=installation_code,
        blob=img_path,
    )

    exists = blob_client.exists()
    if exists:
        logger.info(f"Reference blob found at path: {img_path}")
    else:
        logger.warning(f"Reference blob not found at path: {img_path}")

    return exists


def align_two_images(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Align source image to reference image and transform the polygon accordingly.

    Args:
        reference_image: The reference image (target alignment)
        source_image: The source image to be aligned
        roi_polygon: Polygon points defined in reference image coordinates

    Returns:
        tuple: (warped_polygon, aligned_source_image)
    """
    # Define the polygon points from reference image
    polygon_points = np.array(roi_polygon, dtype=np.float32)

    # Convert images to grayscale
    gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_reference, None)
    keypoints_src, descriptors_src = orb.detectAndCompute(gray_source, None)

    if descriptors_ref is None or descriptors_src is None:
        logger.error("Could not find features in one or both images")
        # Return original source image with original polygon
        return polygon_points.reshape(-1, 1, 2), source_image

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_ref, descriptors_src)

    if len(matches) < 4:
        logger.error(f"Insufficient matches found: {len(matches)}. Need at least 4.")
        return polygon_points.reshape(-1, 1, 2), source_image

    # Sort them in order of distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take only the best matches (top 50% or max 100)
    num_good_matches = min(len(matches), max(len(matches) // 2, 100))
    good_matches = matches[:num_good_matches]

    logger.info(
        f"Using {num_good_matches} good matches out of {len(matches)} total matches"
    )

    # Extract location of good matches
    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_src = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_src[i, :] = keypoints_src[match.trainIdx].pt

    # Calculate homography to transform source image TO reference image coordinates
    H, mask = cv2.findHomography(
        points_src, points_ref, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
    )

    # Check if homography is valid
    if H is None:
        logger.error("Could not compute homography")
        return polygon_points.reshape(-1, 1, 2), source_image

    num_inliers = np.sum(mask) if mask is not None else 0
    logger.info(
        f"Homography computed with {num_inliers} inliers out of {len(good_matches)} matches"
    )

    if num_inliers < 10:
        logger.warning(f"Poor homography quality - only {num_inliers} inliers")
        # Could add fallback behavior here

    # Check if homography is reasonable (not too distorted)
    try:
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 0.1 or abs(det) > 10:
            logger.warning(f"Homography appears distorted (determinant: {det:.3f})")
    except:
        logger.warning("Could not validate homography determinant")

    # Warp source image to align with reference image
    aligned_image = cv2.warpPerspective(
        source_image, H, (reference_image.shape[1], reference_image.shape[0])
    )

    # Since we're aligning source TO reference, the polygon coordinates
    # from the reference image remain valid for the aligned result
    warped_polygon = polygon_points.reshape(-1, 1, 2)

    return cast(NDArray[np.float32], warped_polygon), cast(
        NDArray[np.uint8], aligned_image
    )


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


def create_annotated_thermal_visualization(
    aligned_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    max_temperature: float,
    max_temp_location: tuple[int, int],
    tag_id: str,
    inspection_description: str,
) -> NDArray[np.uint8]:
    """
    Create an annotated visualization of the aligned thermal image with polygon and temperature info.

    Args:
        aligned_image: The aligned thermal image
        polygon_points: Array of polygon vertices
        max_temperature: The maximum temperature value found
        max_temp_location: (x, y) coordinates of the maximum temperature
        tag_id: Tag identifier for the image
        inspection_description: Description of the inspection

    Returns:
        The annotated image array
    """
    # Create a copy of the aligned image for annotation
    annotated_image = aligned_image.copy()

    # Convert polygon points to integer coordinates
    polygon_pts = polygon_points.reshape(-1, 2).astype(np.int32)

    # Draw the polygon outline in bright green
    cv2.polylines(annotated_image, [polygon_pts], True, (0, 255, 0), 3)

    # Draw a circle at the max temperature location in bright red
    cv2.circle(annotated_image, max_temp_location, 8, (0, 0, 255), -1)

    # Add temperature label next to the marker
    temp_label = f"{max_temperature:.1f}"
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.7
    label_thickness = 2

    # Calculate label position (offset from the marker)
    label_x = max_temp_location[0] + 15
    label_y = max_temp_location[1] - 10

    # Get text size for background rectangle
    (label_width, label_height), label_baseline = cv2.getTextSize(
        temp_label, label_font, label_font_scale, label_thickness
    )

    # Draw semi-transparent background for the temperature label
    overlay = annotated_image.copy()
    cv2.rectangle(
        overlay,
        (label_x - 3, label_y - label_height - 3),
        (label_x + label_width + 6, label_y + label_baseline + 3),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

    # Draw the temperature label in bright yellow for high visibility
    cv2.putText(
        annotated_image,
        temp_label,
        (label_x, label_y),
        label_font,
        label_font_scale,
        (0, 255, 255),
        label_thickness,
    )

    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Prepare text information
    temp_text = f"Max Temp: {max_temperature:.1f}"
    location_text = f"Location: ({max_temp_location[0]}, {max_temp_location[1]})"
    tag_text = f"Tag: {tag_id}"
    inspection_text = f"Inspection: {inspection_description}"

    # Calculate text positions (top-left area)
    y_offset = 30
    x_margin = 10

    # Add background rectangles for better text visibility
    texts = [tag_text, inspection_text, temp_text, location_text]

    for i, text in enumerate(texts):
        y_pos = y_offset + i * 35

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw semi-transparent background rectangle
        overlay = annotated_image.copy()
        cv2.rectangle(
            overlay,
            (x_margin - 5, y_pos - text_height - 5),
            (x_margin + text_width + 10, y_pos + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

        # Draw the text in white
        cv2.putText(
            annotated_image,
            text,
            (x_margin, y_pos),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # Add a legend for the colors
    legend_y = annotated_image.shape[0] - 80
    cv2.putText(
        annotated_image, "Legend:", (x_margin, legend_y), font, 0.6, (255, 255, 255), 2
    )
    cv2.putText(
        annotated_image,
        "Green: ROI Polygon",
        (x_margin, legend_y + 25),
        font,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_image,
        "Red: Max Temperature",
        (x_margin, legend_y + 45),
        font,
        0.5,
        (0, 0, 255),
        2,
    )

    return annotated_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get thermal reading inside polygon in image and upload visualization"
    )
    parser.add_argument(
        "--anonymizedBlobStorageLocation",
        required=True,
        help="JSON string for anonymized data blob storage location",
    )
    parser.add_argument(
        "--visualizedBlobStorageLocation",
        required=True,
        help="JSON string for visualized data blob storage location",
    )
    parser.add_argument(
        "--tagId",
        required=True,
        help="JSON string for is break output file",
    )
    parser.add_argument(
        "--inspectionDescription",
        required=True,
        help="JSON string for temperature output file",
    )
    parser.add_argument(
        "--installationCode",
        required=True,
        help="JSON string for installation code",
    )
    parser.add_argument(
        "--temperature-output-file",
        required=False,
        help="JSON string for temperature output file",
        default="/tmp/temperature_output.txt",
    )

    args = parser.parse_args()
    print(f"Arguments received: {args}")
    try:
        anonymized_blob_storage_location = BlobStorageLocation.model_validate(
            json.loads(args.anonymizedBlobStorageLocation)
        )
        visualized_blob_storage_location = BlobStorageLocation.model_validate(
            json.loads(args.visualizedBlobStorageLocation)
        )
        tag_id: str = args.tagId
        inspection_description: str = args.inspectionDescription
        installation_code: str = args.installationCode
        temperature_output_file: str = args.temperature_output_file

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON provided: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing input: {e}")
    if not check_reference_blob_exists(
        tag_id, inspection_description, installation_code
    ):
        logger.error(
            f"Expecting reference image to exist on storage account {REFERENCE_STORAGE_ACCOUNT} for tagId {tag_id} and inspectionDescription {inspection_description} on installationCode {installation_code}"
        )
        return

    reference_image, reference_polygon = load_reference_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    # Download the source image
    source_image_array = download_anonymized_image(anonymized_blob_storage_location)

    warped_polygon, aligned_image = align_two_images(
        reference_image,
        source_image_array,
        reference_polygon,
    )

    # Find the maximum temperature within the polygon region
    max_temperature, max_temp_location = find_max_temperature_in_polygon(
        aligned_image, warped_polygon
    )

    logger.info(
        f"Maximum temperature found: {max_temperature} at location {max_temp_location}"
    )

    # Create annotated visualization
    annotated_image = create_annotated_thermal_visualization(
        aligned_image,
        warped_polygon,
        max_temperature,
        max_temp_location,
        tag_id,
        inspection_description,
    )

    logger.info(f"Created annotated thermal visualization")

    upload_to_visualized(
        visualized_blob_storage_location,
        annotated_image,
    )

    with open(temperature_output_file, "w") as file:
        file.write(str(max_temperature))
        print(
            f"Max temperature: {max_temperature} written to {temperature_output_file}"
        )


if __name__ == "__main__":
    main()
