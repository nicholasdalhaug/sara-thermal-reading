from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# Constants for visual styling
POLYGON_COLOR = (128, 128, 128)  # Grey
MAX_TEMP_MARKER_COLOR = (0, 0, 0)  # Black
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)
BACKGROUND_COLOR_BLACK = (0, 0, 0)
BACKGROUND_COLOR_WHITE = (255, 255, 255)
OVERLAY_ALPHA = 0.7

# Font constants
FONT_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCRIPT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX


def _scale(value: float, scale_factor: float) -> int:
    """Helper to scale a value and convert to int."""
    return int(value * scale_factor)


def _draw_text_with_background(
    image: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    font: int,
    font_scale: float,
    text_color: tuple[int, int, int],
    thickness: int,
    bg_color: tuple[int, int, int] = BACKGROUND_COLOR_BLACK,
    bg_alpha: float = OVERLAY_ALPHA,
    padding: int = 5,
) -> None:
    """Draw text with a semi-transparent background rectangle."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position

    # Draw background
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.addWeighted(overlay, bg_alpha, image, 1 - bg_alpha, 0, image)

    # Draw text
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)


def _draw_legend(
    image: NDArray[np.uint8],
    scale_factor: float,
    x_margin: int,
) -> None:
    """Draw the legend showing polygon and max temperature markers."""
    legend_y = _scale(image.shape[0] / scale_factor - 80, scale_factor)
    font = FONT_SIMPLEX
    font_scale = 0.5 * scale_factor
    thickness = _scale(2, scale_factor)

    legend_items = ["Grey: ROI", "Black: Max Temperature"]

    # Calculate max width for background
    max_width = max(
        cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        for text in legend_items
    )

    # Draw background
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (
            _scale(x_margin / scale_factor - 5, scale_factor),
            _scale(legend_y / scale_factor + 10, scale_factor),
        ),
        (
            _scale((x_margin + max_width) / scale_factor + 10, scale_factor),
            _scale(legend_y / scale_factor + 60, scale_factor),
        ),
        BACKGROUND_COLOR_BLACK,
        -1,
    )
    cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0, image)

    # Draw legend items
    for i, text in enumerate(legend_items):
        y_pos = _scale(legend_y / scale_factor + 25 + i * 20, scale_factor)
        cv2.putText(
            image,
            text,
            (x_margin, y_pos),
            font,
            font_scale,
            TEXT_COLOR_WHITE,
            thickness,
        )


def _create_colorbar(
    annotated_image: NDArray[np.uint8],
    thermal_data: NDArray[np.float64],
    scale_factor: float,
) -> NDArray[np.uint8]:
    """Create and attach a colorbar to the right side of the image."""
    min_temp = float(np.min(thermal_data))
    max_temp = float(np.max(thermal_data))

    # Colorbar dimensions
    colorbar_width = _scale(40, scale_factor)
    colorbar_height = _scale(300, scale_factor)
    colorbar_margin = _scale(20, scale_factor)
    label_space = _scale(80, scale_factor)

    # Create canvas with white background
    canvas_width = annotated_image.shape[1] + colorbar_width + label_space
    canvas = np.full((annotated_image.shape[0], canvas_width, 3), 255, dtype=np.uint8)

    # Copy annotated image to left side
    canvas[:, : annotated_image.shape[1]] = annotated_image

    # Create colorbar gradient
    gradient = np.linspace(255, 0, colorbar_height, dtype=np.uint8)
    gradient = np.tile(gradient.reshape(-1, 1), (1, colorbar_width))
    colorbar_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    # Position colorbar
    y_start = (canvas.shape[0] - colorbar_height) // 2
    x_start = annotated_image.shape[1] + colorbar_margin

    # Place colorbar on canvas
    canvas[y_start : y_start + colorbar_height, x_start : x_start + colorbar_width] = (
        colorbar_colored
    )

    # Draw border around colorbar
    cv2.rectangle(
        canvas,
        (x_start, y_start),
        (x_start + colorbar_width, y_start + colorbar_height),
        TEXT_COLOR_WHITE,
        _scale(2, scale_factor),
    )

    # Add temperature labels
    font_scale = 0.4 * scale_factor
    thickness = _scale(1, scale_factor)
    label_x = x_start + colorbar_width + _scale(5, scale_factor)

    temp_labels = [
        (max_temp, y_start + _scale(15, scale_factor)),
        (
            (min_temp + max_temp) / 2,
            y_start + colorbar_height // 2 + _scale(5, scale_factor),
        ),
        (min_temp, y_start + colorbar_height - _scale(5, scale_factor)),
    ]

    for temp, y_pos in temp_labels:
        cv2.putText(
            canvas,
            f"{temp:.1f}",
            (label_x, y_pos),
            FONT_SIMPLEX,
            font_scale,
            TEXT_COLOR_BLACK,
            thickness,
        )

    return canvas


def create_annotated_thermal_visualization(
    aligned_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    max_temperature: float,
    max_temp_location: tuple[int, int],
    tag_id: str,
    inspection_description: str,
    scale_factor: float = 2.0,
    show_colorbar: bool = True,
    thermal_data: NDArray[np.float64] | None = None,
) -> NDArray[np.uint8]:
    """
    Create an annotated visualization of the aligned thermal image with polygon and temperature info.

    Args:
        aligned_image: The aligned thermal image (uint8, normalized)
        polygon_points: Array of polygon vertices
        max_temperature: The maximum temperature value found
        max_temp_location: (x, y) coordinates of the maximum temperature
        tag_id: Tag identifier for the image
        inspection_description: Description of the inspection
        scale_factor: Factor to upscale the image for sharper text rendering (default: 2.0)
        show_colorbar: Whether to show temperature colorbar on the right side (default: True)
        thermal_data: Optional original thermal data array (float64) for accurate temperature mapping

    Returns:
        The annotated image array
    """
    # Prepare image
    upscaled_shape = (
        _scale(aligned_image.shape[1], scale_factor),
        _scale(aligned_image.shape[0], scale_factor),
    )

    # Apply colormap to grayscale images
    if len(aligned_image.shape) == 2:
        aligned_image = cast(
            NDArray[np.uint8], cv2.applyColorMap(aligned_image, cv2.COLORMAP_JET)
        )

    annotated_image = cast(
        NDArray[np.uint8],
        cv2.resize(aligned_image, upscaled_shape, interpolation=cv2.INTER_CUBIC),
    )

    # Scale coordinates
    scaled_polygon_points = polygon_points * scale_factor
    scaled_max_temp_location = (
        _scale(max_temp_location[0], scale_factor),
        _scale(max_temp_location[1], scale_factor),
    )
    polygon_pts = scaled_polygon_points.reshape(-1, 2).astype(np.int32)

    # Draw polygon
    cv2.polylines(
        annotated_image,
        [polygon_pts],
        True,
        POLYGON_COLOR,
        _scale(3, scale_factor),
    )

    # Draw max temperature marker
    cv2.circle(
        annotated_image,
        scaled_max_temp_location,
        _scale(8, scale_factor),
        MAX_TEMP_MARKER_COLOR,
        -1,
    )

    # Draw temperature label next to marker
    temp_label = f"{max_temperature:.1f}"
    label_position = (
        _scale(scaled_max_temp_location[0] / scale_factor + 15, scale_factor),
        _scale(scaled_max_temp_location[1] / scale_factor - 10, scale_factor),
    )

    _draw_text_with_background(
        annotated_image,
        temp_label,
        label_position,
        FONT_SCRIPT,
        0.5 * scale_factor,
        TEXT_COLOR_WHITE,
        _scale(2, scale_factor),
        BACKGROUND_COLOR_BLACK,
        OVERLAY_ALPHA,
        _scale(3, scale_factor),
    )

    # Draw legend
    x_margin = _scale(10, scale_factor)
    _draw_legend(annotated_image, scale_factor, x_margin)

    # Add colorbar if requested
    if show_colorbar and thermal_data is not None:
        annotated_image = _create_colorbar(annotated_image, thermal_data, scale_factor)

    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        annotated_image = cast(
            NDArray[np.uint8], cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        )
    return annotated_image
