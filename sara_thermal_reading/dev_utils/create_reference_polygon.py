import json
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np
from loguru import logger

from sara_thermal_reading.file_io.fff_loader import load_fff


def create_reference_polygon(
    fff_image_path: Path, polygon_json_output_path: Path
) -> None:
    """
    Opens an interactive window to draw a polygon on the thermal image.
    Saves the polygon coordinates to a JSON file.
    """
    if not fff_image_path.exists():
        logger.error(f"Image file not found: {fff_image_path}")
        return

    thermal_image = load_fff(str(fff_image_path))

    norm_image = np.zeros_like(thermal_image)
    cv2.normalize(thermal_image, norm_image, 0, 255, cv2.NORM_MINMAX)
    display_image_gray = norm_image.astype(np.uint8)

    base_image = cv2.applyColorMap(display_image_gray, cv2.COLORMAP_JET)
    display_image = base_image.copy()

    points: List[List[int]] = []
    window_name = "Draw Polygon (Click to add points, Enter to save)"

    def draw_dotted_line(
        img: np.ndarray,
        pt1: Sequence[int],
        pt2: Sequence[int],
        color: tuple[int, int, int],
        thickness: int = 1,
        gap: int = 10,
    ) -> None:
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0:
            return
        pts = np.linspace(pt1, pt2, int(dist / gap) + 1)
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(
                    img,
                    tuple(pts[i].astype(int)),
                    tuple(pts[i + 1].astype(int)),
                    color,
                    thickness,
                )

    def redraw() -> None:
        nonlocal display_image
        display_image = base_image.copy()
        for i, point in enumerate(points):
            cv2.circle(display_image, tuple(point), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(
                    display_image, tuple(points[i - 1]), tuple(point), (0, 255, 0), 2
                )

        if len(points) > 2:
            draw_dotted_line(
                display_image,
                tuple(points[-1]),
                tuple(points[0]),
                (0, 255, 0),
                thickness=1,
                gap=5,
            )

        cv2.imshow(window_name, display_image)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if len(points) < 3:
                logger.warning("Polygon must have at least 3 points.")
                continue

            # Close the polygon visually
            cv2.line(display_image, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow(window_name, display_image)
            cv2.waitKey(500)  # Show closed polygon briefly
            break
        elif key == 27:  # Esc
            logger.info("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    with open(polygon_json_output_path, "w") as f:
        json.dump(points, f, indent=4)

    logger.info(f"Polygon saved to {polygon_json_output_path}")
    print(json.dumps(points, indent=4))
