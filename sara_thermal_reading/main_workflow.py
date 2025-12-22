import json
from typing import cast

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.file_io.blob import BlobStorageLocation
from sara_thermal_reading.logger import setup_logger

setup_logger()
from loguru import logger

from sara_thermal_reading.file_io.file_utils import (
    check_reference_blob_exists,
    download_anonymized_image,
    load_reference_image_and_polygon,
    upload_to_visualized,
)
from sara_thermal_reading.find_max_temperature_in_polygon import (
    find_max_temperature_in_polygon,
)
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)


def run_thermal_reading_workflow(
    anonymized_blob_storage_location: BlobStorageLocation,
    visualized_blob_storage_location: BlobStorageLocation,
    tag_id: str,
    inspection_description: str,
    installation_code: str,
    temperature_output_file: str,
) -> None:
    if not check_reference_blob_exists(
        tag_id, inspection_description, installation_code
    ):
        logger.error(
            f"Expecting reference image to exist on storage account {settings.REFERENCE_STORAGE_ACCOUNT} for tagId {tag_id} and inspectionDescription {inspection_description} on installationCode {installation_code}"
        )
        return

    reference_image, reference_polygon = load_reference_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    # Download the source image
    source_image_array = download_anonymized_image(anonymized_blob_storage_location)

    warped_polygon, aligned_image = align_two_images_orb_bf_cv2(
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
