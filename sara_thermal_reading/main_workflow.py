import numpy as np
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
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)
from sara_thermal_reading.image_processing.find_max_temperature_in_polygon import (
    find_max_temperature_in_polygon,
)
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)


def process_thermal_image(
    reference_image: NDArray[np.uint8],
    source_image_array: NDArray[np.uint8],
    reference_polygon: list[tuple[int, int]],
    tag_id: str,
    inspection_description: str,
) -> tuple[float, NDArray[np.uint8]]:

    warped_polygon, aligned_image = align_two_images_orb_bf_cv2(
        reference_image,
        source_image_array,
        reference_polygon,
    )

    max_temperature, max_temp_location = find_max_temperature_in_polygon(
        aligned_image, warped_polygon
    )

    logger.info(
        f"Maximum temperature found: {max_temperature} at location {max_temp_location}"
    )

    annotated_image = create_annotated_thermal_visualization(
        aligned_image,
        warped_polygon,
        max_temperature,
        max_temp_location,
        tag_id,
        inspection_description,
    )

    return max_temperature, annotated_image


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

    source_image_array = download_anonymized_image(anonymized_blob_storage_location)

    max_temperature, annotated_image = process_thermal_image(
        reference_image,
        source_image_array,
        reference_polygon,
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
