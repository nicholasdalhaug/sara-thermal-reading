from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from numpy.typing import NDArray

from sara_thermal_reading.main import (
    BlobStorageLocation,
    align_two_images,
    create_annotated_thermal_visualization,
    download_anonymized_image,
    find_max_temperature_in_polygon,
    load_reference_image_and_polygon,
)

load_dotenv()


def detect_temperature_range(thermal_image: NDArray[np.uint8]) -> tuple[float, float]:
    """
    Attempt to detect the temperature range from a thermal image.

    For now, returns a reasonable default range for typical thermal cameras.
    In a production system, this could analyze the image metadata or
    temperature scale bars to determine the actual range.

    Args:
        thermal_image: The thermal image array

    Returns:
        Tuple of (min_temp, max_temp) in Celsius
    """
    # Default temperature ranges for common thermal imaging scenarios
    # These could be made configurable or detected from image metadata

    # Analyze the image to make a better guess
    if len(thermal_image.shape) == 3:
        gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = thermal_image

    # Get basic statistics
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)

    logger.info(
        f"Image statistics - Min: {min_val}, Max: {max_val}, Mean: {mean_val:.1f}, Std: {std_val:.1f}"
    )

    # Based on typical thermal imaging applications
    if mean_val > 150:  # Likely high-temperature application
        return (50.0, 150.0)
    elif mean_val < 50:  # Likely low-temperature application
        return (10.0, 40.0)
    else:  # Medium range application (most common)
        return (20.0, 100.0)


def save_alignment_results(
    image1: NDArray[np.uint8],
    image2: NDArray[np.uint8],
    aligned_image1: NDArray[np.uint8],
    warped_polygon: NDArray[np.float32],
    save_dir: str = "results",
) -> dict[str, str]:
    """
    Save alignment results including reference, aligned, and overlay images.

    Args:
        image1: Reference image
        image2: Source image
        aligned_image1: Aligned version of image1
        warped_polygon: Warped polygon points
        save_dir: Directory to save results

    Returns:
        Dictionary containing paths to saved files
    """
    alignment_dir = Path(save_dir)
    alignment_dir.mkdir(exist_ok=True)

    saved_files = {}

    # Save reference image (image1)
    reference_path = alignment_dir / "reference_image.jpg"
    cv2.imwrite(str(reference_path), image1)
    saved_files["reference"] = str(reference_path)
    logger.info(f"Saved reference image to {reference_path}")

    # Save aligned source image
    aligned_path = alignment_dir / "aligned_source_image.jpg"
    cv2.imwrite(str(aligned_path), aligned_image1)
    saved_files["aligned"] = str(aligned_path)
    logger.info(f"Saved aligned source image to {aligned_path}")

    # Create overlay visualization for better comparison
    overlay = cv2.addWeighted(image1, 0.5, aligned_image1, 0.5, 0)
    overlay_path = alignment_dir / "overlay_comparison.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    saved_files["overlay"] = str(overlay_path)
    logger.info(f"Saved overlay comparison to {overlay_path}")

    # Draw polygon on images for visualization
    image1_with_polygon = image1.copy()
    aligned_with_polygon = aligned_image1.copy()
    polygon_pts = warped_polygon.reshape(-1, 2).astype(np.int32)

    cv2.polylines(image1_with_polygon, [polygon_pts], True, (0, 255, 0), 3)
    cv2.polylines(aligned_with_polygon, [polygon_pts], True, (0, 255, 0), 3)

    reference_polygon_path = alignment_dir / "reference_with_polygon.jpg"
    aligned_polygon_path = alignment_dir / "aligned_with_polygon.jpg"

    cv2.imwrite(str(reference_polygon_path), image1_with_polygon)
    cv2.imwrite(str(aligned_polygon_path), aligned_with_polygon)
    saved_files["reference_with_polygon"] = str(reference_polygon_path)
    saved_files["aligned_with_polygon"] = str(aligned_polygon_path)
    logger.info(
        f"Saved polygon visualizations to {reference_polygon_path} and {aligned_polygon_path}"
    )

    return saved_files


def save_temperature_analysis(
    thermal_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    max_temp_location: tuple[int, int],
    save_dir: str = "results",
) -> dict[str, str]:
    """
    Save temperature analysis visualizations.

    Args:
        thermal_image: The thermal image array
        polygon_points: Array of polygon vertices
        max_temp_location: (x, y) coordinates of maximum temperature
        save_dir: Directory to save results

    Returns:
        Dictionary containing paths to saved files
    """
    alignment_dir = Path(save_dir)
    alignment_dir.mkdir(exist_ok=True)

    saved_files = {}

    # Convert polygon points to integer coordinates
    polygon_pts = polygon_points.reshape(-1, 2).astype(np.int32)

    # Create visualization showing the polygon mask and max temperature location
    visualization = thermal_image.copy()
    cv2.polylines(visualization, [polygon_pts], True, (0, 255, 0), 2)
    cv2.circle(
        visualization, max_temp_location, 5, (0, 0, 255), -1
    )  # Red circle at max temp location

    mask_viz_path = alignment_dir / "temperature_analysis.jpg"
    cv2.imwrite(str(mask_viz_path), visualization)
    saved_files["temperature_analysis"] = str(mask_viz_path)
    logger.info(f"Saved temperature analysis visualization to {mask_viz_path}")

    # Create mask for the polygon region and save masked thermal region
    height, width = thermal_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)

    # Convert thermal image to grayscale if it's in color
    if len(thermal_image.shape) == 3:
        gray_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_thermal = thermal_image

    # Apply mask to get only the polygon region
    masked_thermal = cv2.bitwise_and(gray_thermal, gray_thermal, mask=mask)

    masked_viz_path = alignment_dir / "masked_thermal_region.jpg"
    cv2.imwrite(str(masked_viz_path), masked_thermal)
    saved_files["masked_thermal"] = str(masked_viz_path)
    logger.info(f"Saved masked thermal region to {masked_viz_path}")

    return saved_files


def save_annotated_thermal_visualization(
    annotated_image: NDArray[np.uint8],
    tag_id: str,
    inspection_description: str,
    save_dir: str = "results",
) -> str:
    """
    Save the annotated thermal visualization image.

    Args:
        annotated_image: The annotated thermal image
        tag_id: Tag identifier for the image
        inspection_description: Description of the inspection
        save_dir: Directory to save results

    Returns:
        Path to the saved annotated image
    """
    results_dir = Path(save_dir)
    results_dir.mkdir(exist_ok=True)

    annotated_path = (
        results_dir / f"annotated_thermal_{tag_id}_{inspection_description}.jpg"
    )
    cv2.imwrite(str(annotated_path), annotated_image)

    logger.info(f"Saved annotated thermal visualization to {annotated_path}")

    return str(annotated_path)


def example_thermal_processing_with_file_saving():
    """
    Example function that demonstrates the complete thermal processing workflow
    with local file saving.
    """
    # These would typically come from command line arguments or function parameters
    tag_id = "52-LQ-5254"
    inspection_description = "Steam-trap"
    installation_code = "kaa"

    try:
        print(f"Running thermal processing example with file saving:")
        print(f"  Tag ID: {tag_id}")
        print(f"  Inspection: {inspection_description}")
        print(f"  Installation: {installation_code}")

        # Create save directory
        save_dir = "results"
        results_dir = Path(save_dir)
        results_dir.mkdir(exist_ok=True)

        # Example blob storage location (this would come from actual parameters)
        source_blob_location = BlobStorageLocation(
            blobContainer="test",
            blobName="test_steam_trap/52-LQ-5254__ThermalImage__Steam-trap__20251204-125407.jpeg",
        )

        # Load reference image and polygon
        reference_image, reference_polygon = load_reference_image_and_polygon(
            installation_code, tag_id, inspection_description
        )

        # Download the source image
        source_image_array = download_anonymized_image(source_blob_location)

        # 1. Save reference image
        reference_path = results_dir / "01_reference_image.jpg"
        cv2.imwrite(str(reference_path), reference_image)
        logger.info(f"Saved reference image to {reference_path}")

        # 2. Save source image
        source_path = results_dir / "02_source_image.jpg"
        cv2.imwrite(str(source_path), source_image_array)
        logger.info(f"Saved source image to {source_path}")

        # 3. Save reference image with polygon
        reference_with_polygon = reference_image.copy()
        polygon_pts = np.array(reference_polygon, dtype=np.int32)

        # Draw filled polygon with transparency
        overlay = reference_with_polygon.copy()
        cv2.fillPoly(overlay, [polygon_pts], (0, 255, 0))  # Green fill
        reference_with_polygon = cv2.addWeighted(
            reference_with_polygon, 0.7, overlay, 0.3, 0
        )

        # Draw polygon outline
        cv2.polylines(
            reference_with_polygon, [polygon_pts], True, (0, 255, 0), 3
        )  # Green outline

        reference_polygon_path = results_dir / "03_reference_with_polygon.jpg"
        cv2.imwrite(str(reference_polygon_path), reference_with_polygon)
        logger.info(f"Saved reference image with polygon to {reference_polygon_path}")

        # Perform alignment
        warped_polygon, aligned_image = align_two_images(
            reference_image,
            source_image_array,
            reference_polygon,
        )

        # 4. Save alignment comparison (overlay of reference and aligned source)
        alignment_overlay = cv2.addWeighted(reference_image, 0.5, aligned_image, 0.5, 0)
        alignment_path = results_dir / "04_alignment_overlay.jpg"
        cv2.imwrite(str(alignment_path), alignment_overlay)
        logger.info(f"Saved alignment overlay to {alignment_path}")

        # Detect temperature range from the thermal image
        temp_range = detect_temperature_range(aligned_image)
        logger.info(
            f"Detected temperature range: {temp_range[0]:.1f}°C to {temp_range[1]:.1f}°C"
        )

        # Find maximum temperature with proper temperature mapping
        max_temperature, max_temp_location = find_max_temperature_in_polygon(
            aligned_image, warped_polygon, temp_range
        )

        # 5. Save aligned source image with polygon and temperature marking
        aligned_with_annotations = aligned_image.copy()
        warped_polygon_pts = warped_polygon.reshape(-1, 2).astype(np.int32)

        # Draw filled polygon with transparency
        overlay_aligned = aligned_with_annotations.copy()
        cv2.fillPoly(overlay_aligned, [warped_polygon_pts], (0, 255, 0))  # Green fill
        aligned_with_annotations = cv2.addWeighted(
            aligned_with_annotations, 0.7, overlay_aligned, 0.3, 0
        )

        # Draw polygon outline
        cv2.polylines(
            aligned_with_annotations, [warped_polygon_pts], True, (0, 255, 0), 3
        )

        # Mark maximum temperature location
        cv2.circle(
            aligned_with_annotations, max_temp_location, 10, (0, 0, 255), -1
        )  # Red circle
        cv2.circle(
            aligned_with_annotations, max_temp_location, 15, (255, 255, 255), 3
        )  # White outline

        # Add temperature label
        label_text = f"Max Temp: {max_temperature:.1f}°C"
        label_position = (max_temp_location[0] + 20, max_temp_location[1] - 20)

        # Add background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            aligned_with_annotations,
            (label_position[0] - 5, label_position[1] - text_height - 5),
            (label_position[0] + text_width + 5, label_position[1] + baseline + 5),
            (0, 0, 0),
            -1,
        )  # Black background

        # Add text
        cv2.putText(
            aligned_with_annotations,
            label_text,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )  # White text

        aligned_annotated_path = (
            results_dir / "05_aligned_source_with_polygon_and_temperature.jpg"
        )
        cv2.imwrite(str(aligned_annotated_path), aligned_with_annotations)
        logger.info(
            f"Saved aligned source with polygon and temperature to {aligned_annotated_path}"
        )

        print(f"\nProcessing complete! Maximum temperature: {max_temperature:.1f}°C")
        print(f"Temperature location: {max_temp_location}")
        print("\nSaved files:")
        print(f"  1. Reference image: {reference_path}")
        print(f"  2. Source image: {source_path}")
        print(f"  3. Reference with polygon: {reference_polygon_path}")
        print(f"  4. Alignment overlay: {alignment_path}")
        print(f"  5. Aligned source with annotations: {aligned_annotated_path}")
        print(f"\nCheck the '{results_dir}' folder for all saved files.")

    except Exception as e:
        print(f"Error during thermal processing: {e}")
        print(
            "Make sure your environment variables are set and the reference data exists."
        )


def save_reference_image_with_polygon_locally(
    tag_id: str, inspection_description: str, installation_code: str | None = None
) -> str:

    image_array, polygon = load_reference_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    # Convert image array to OpenCV format (BGR)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # RGBA to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:
        # Grayscale to BGR
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    # Convert polygon points to numpy array
    polygon_points = np.array(polygon, dtype=np.int32)

    # Draw polygon on image
    # Draw filled polygon with transparency
    overlay = image_bgr.copy()
    cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))  # Green fill
    image_with_polygon = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

    # Draw polygon outline
    cv2.polylines(
        image_with_polygon, [polygon_points], True, (0, 255, 0), 3
    )  # Green outline

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    filename = f"{tag_id}_{inspection_description}_reference_with_polygon.png"

    output_path = results_dir / filename

    # Save image as PNG
    success = cv2.imwrite(str(output_path), image_with_polygon)

    if success:
        logger.info(f"Saved reference image with polygon to: {output_path}")
        return str(output_path)
    else:
        raise RuntimeError(f"Failed to save image to {output_path}")


if __name__ == "__main__":
    example_thermal_processing_with_file_saving()
