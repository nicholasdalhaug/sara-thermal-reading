import logging

import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings

logger = logging.getLogger(__name__)

from sara_thermal_reading.file_io.blob import BlobStorageLocation
from sara_thermal_reading.file_io.file_utils import (
    check_reference_blob_exists,
    download_anonymized_fff_image,
    load_reference_fff_image_and_polygon,
    upload_to_visualized,
)
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)
from sara_thermal_reading.image_processing.convert_thermal_to_uint8 import (
    convert_thermal_to_uint8,
)
from sara_thermal_reading.image_processing.find_max_temperature_in_polygon_raw_thermal import (
    find_max_temperature_in_polygon_raw_thermal,
)
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)


def process_thermal_image_fff(
    reference_image: NDArray[np.float64],
    source_image_array: NDArray[np.float64],
    reference_polygon: list[tuple[int, int]],
    tag_id: str,
    inspection_description: str,
) -> tuple[
    float,
    tuple[int, int],
    NDArray[np.uint8],
    list[tuple[int, int]],
    NDArray[np.uint8],
]:

    reference_image_uint8 = convert_thermal_to_uint8(reference_image)
    source_image_uint8 = convert_thermal_to_uint8(source_image_array)

    warped_polygon_list, warped_reference_img = align_two_images_orb_bf_cv2(
        reference_image_uint8,
        source_image_uint8,
        reference_polygon,
    )

    # Convert list of tuples back to numpy array for processing functions
    warped_polygon_array = np.array(warped_polygon_list, dtype=np.float32)

    max_temperature, max_temp_location = find_max_temperature_in_polygon_raw_thermal(
        source_image_array, warped_polygon_array
    )

    logger.info(
        f"Maximum temperature found: {max_temperature} at location {max_temp_location}"
    )

    annotated_image = create_annotated_thermal_visualization(
        source_image_uint8,
        warped_polygon_array,
        max_temperature,
        max_temp_location,
        tag_id,
        inspection_description,
    )

    return (
        max_temperature,
        max_temp_location,
        annotated_image,
        warped_polygon_list,
        warped_reference_img,
    )


def run_thermal_reading_fff_workflow(
    anonymized_blob_storage_location: BlobStorageLocation,
    visualized_blob_storage_location: BlobStorageLocation,
    tag_id: str,
    inspection_description: str,
    installation_code: str,
    temperature_output_file: str,
) -> None:

    logger.info(f"Starting run thermal reading fff workflow")

    if not check_reference_blob_exists(
        tag_id, inspection_description, installation_code
    ):
        logger.error(
            f"Expecting reference image to exist on storage account {installation_code} for tagId {tag_id} and inspectionDescription {inspection_description}"
        )
        return

    logger.info(f"Loading reference image and polygon")
    reference_image, reference_polygon = load_reference_fff_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    logger.info(f"Loading source polygon")
    source_image_array = download_anonymized_fff_image(anonymized_blob_storage_location)

    logger.info(f"Processing thermal fff image")
    max_temperature, _, annotated_image, _, _ = process_thermal_image_fff(
        reference_image,
        source_image_array,
        reference_polygon,
        tag_id,
        inspection_description,
    )

    logger.info(f"Created annotated thermal visualization")
    logger.info("Uploading to visualized")
    upload_to_visualized(
        visualized_blob_storage_location,
        annotated_image,
    )

    with open(temperature_output_file, "w") as file:
        file.write(str(max_temperature))
        logger.info(
            f"Max temperature: {max_temperature} written to {temperature_output_file}"
        )
