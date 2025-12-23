import json

import numpy as np
from azure.storage.blob import BlobServiceClient
from loguru import logger
from numpy.typing import NDArray
from PIL import Image

from sara_thermal_reading.config.settings import settings

from .blob import (
    BlobStorageLocation,
    download_blob_to_bytes,
    download_blob_to_image,
    download_blob_to_json,
    upload_image_to_blob,
)
from .fff_loader import load_fff_from_bytes


def download_anonymized_image(
    anonymized_blob_storage_location: BlobStorageLocation,
) -> NDArray[np.uint8]:
    logger.info(f"Processing new thermal image")
    src_blob_service_client = BlobServiceClient.from_connection_string(
        settings.SOURCE_STORAGE_CONNECTION_STRING
    )

    anonymized_image_array: NDArray[np.uint8] = download_blob_to_image(
        src_blob_service_client, anonymized_blob_storage_location
    )
    logger.info(
        f"Downloaded image from source storage account, shape: {anonymized_image_array.shape}"
    )

    return anonymized_image_array


def download_fff_image(
    blob_service_client: BlobServiceClient, blob_storage_location: BlobStorageLocation
) -> NDArray[np.float64]:
    blob_bytes = download_blob_to_bytes(blob_service_client, blob_storage_location)
    thermal_image_array = load_fff_from_bytes(blob_bytes)
    return thermal_image_array


def download_anonymized_fff_image(
    anonymized_blob_storage_location: BlobStorageLocation,
) -> NDArray[np.float64]:
    logger.info(f"Processing new thermal FFF image")

    src_blob_service_client = BlobServiceClient.from_connection_string(
        settings.SOURCE_STORAGE_CONNECTION_STRING
    )

    thermal_image_array = download_fff_image(
        src_blob_service_client, anonymized_blob_storage_location
    )

    logger.info(
        f"Downloaded FFF image from source storage account, shape: {thermal_image_array.shape}"
    )

    return thermal_image_array


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def upload_to_visualized(
    visualized_blob_storage_location: BlobStorageLocation,
    image: NDArray[np.uint8],
) -> None:
    logger.info(f"Uploading annotated thermal image to visualized storage account")
    vis_blob_service_client = BlobServiceClient.from_connection_string(
        settings.DESTINATION_STORAGE_CONNECTION_STRING
    )

    upload_image_to_blob(
        vis_blob_service_client, visualized_blob_storage_location, image
    )
    logger.info(
        f"Uploaded annotated thermal image to visualized storage account: {visualized_blob_storage_location.blob_container}/{visualized_blob_storage_location.blob_name}"
    )


def load_reference_polygon(
    installation_code: str, tag_id: str, inspection_description: str
) -> list[tuple[int, int]]:
    try:
        ref_blob_service_client = BlobServiceClient.from_connection_string(
            settings.REFERENCE_STORAGE_CONNECTION_STRING
        )
        polygon_path = f"{tag_id}_{inspection_description}/reference_polygon.json"
        polygon_json = download_blob_to_json(
            ref_blob_service_client,
            BlobStorageLocation(blobContainer=installation_code, blobName=polygon_path),
        )
        polygon = json.loads(polygon_json)
        return polygon
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode polygon JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading reference polygon: {e}")
        raise


def load_reference_image_and_polygon(
    installation_code: str, tag_id: str, inspection_description: str
) -> tuple[NDArray[np.uint8], list[tuple[int, int]]]:
    try:
        ref_blob_service_client = BlobServiceClient.from_connection_string(
            settings.REFERENCE_STORAGE_CONNECTION_STRING
        )
        img_path = (
            f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_FILENAME}"
        )
        image_array = download_blob_to_image(
            ref_blob_service_client,
            BlobStorageLocation(blobContainer=installation_code, blobName=img_path),
        )
        polygon = load_reference_polygon(
            installation_code, tag_id, inspection_description
        )
    except Exception as e:
        logger.error(f"Error loading reference image and polygon: {e}")
        raise
    return image_array, polygon


def load_reference_fff_image_and_polygon(
    installation_code: str, tag_id: str, inspection_description: str
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    try:
        ref_blob_service_client = BlobServiceClient.from_connection_string(
            settings.REFERENCE_STORAGE_CONNECTION_STRING
        )

        img_path = (
            f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_FILENAME}"
        )

        image_array = download_fff_image(
            ref_blob_service_client,
            BlobStorageLocation(blobContainer=installation_code, blobName=img_path),
        )

        polygon = load_reference_polygon(
            installation_code, tag_id, inspection_description
        )
    except Exception as e:
        logger.error(f"Error loading reference FFF image and polygon: {e}")
        raise
    return image_array, polygon


def check_reference_blob_exists(
    tag_id: str, inspection_description: str, installation_code: str
) -> bool:
    logger.info(
        f"Checking if reference blob exists for tag_id: {tag_id}, inspection_description: {inspection_description}, installation_code: {installation_code}"
    )

    ref_blob_service_client = BlobServiceClient.from_connection_string(
        settings.REFERENCE_STORAGE_CONNECTION_STRING
    )
    img_path = f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_FILENAME}"
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
