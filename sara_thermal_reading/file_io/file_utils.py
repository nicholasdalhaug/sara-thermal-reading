import json
import os

import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from loguru import logger
from numpy.typing import NDArray
from PIL import Image

from .blob import (
    BlobStorageLocation,
    download_blob_to_image,
    download_blob_to_json,
    upload_image_to_blob,
)

load_dotenv()


def get_env_var_or_raise_error(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set")
    return value


SOURCE_STORAGE_ACCOUNT = get_env_var_or_raise_error("SOURCE_STORAGE_ACCOUNT")
SOURCE_STORAGE_CONNECTION_STRING = get_env_var_or_raise_error(
    "SOURCE_STORAGE_CONNECTION_STRING"
)
DESTINATION_STORAGE_ACCOUNT = get_env_var_or_raise_error("DESTINATION_STORAGE_ACCOUNT")
DESTINATION_STORAGE_CONNECTION_STRING = get_env_var_or_raise_error(
    "DESTINATION_STORAGE_CONNECTION_STRING"
)
REFERENCE_STORAGE_ACCOUNT = get_env_var_or_raise_error("REFERENCE_STORAGE_ACCOUNT")
REFERENCE_STORAGE_CONNECTION_STRING = get_env_var_or_raise_error(
    "REFERENCE_STORAGE_CONNECTION_STRING"
)


def download_anonymized_image(
    anonymized_blob_storage_location: BlobStorageLocation,
) -> NDArray[np.uint8]:
    logger.info(f"Processing new thermal image")
    src_blob_service_client = BlobServiceClient.from_connection_string(
        SOURCE_STORAGE_CONNECTION_STRING
    )

    anonymized_image_array: NDArray[np.uint8] = download_blob_to_image(
        src_blob_service_client, anonymized_blob_storage_location
    )
    logger.info(
        f"Downloaded image from source storage account, shape: {anonymized_image_array.shape}"
    )

    return anonymized_image_array


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def upload_to_visualized(
    visualized_blob_storage_location: BlobStorageLocation,
    image: NDArray[np.uint8],
) -> None:
    logger.info(f"Uploading annotated thermal image to visualized storage account")
    vis_blob_service_client = BlobServiceClient.from_connection_string(
        DESTINATION_STORAGE_CONNECTION_STRING
    )

    upload_image_to_blob(
        vis_blob_service_client, visualized_blob_storage_location, image
    )
    logger.info(
        f"Uploaded annotated thermal image to visualized storage account: {visualized_blob_storage_location.blob_container}/{visualized_blob_storage_location.blob_name}"
    )


def load_reference_image_and_polygon(
    installation_code: str, tag_id: str, inspection_description: str
) -> tuple[NDArray[np.uint8], list[tuple[int, int]]]:
    try:
        ref_blob_service_client = BlobServiceClient.from_connection_string(
            REFERENCE_STORAGE_CONNECTION_STRING
        )
        img_path = f"{tag_id}_{inspection_description}/reference_image.jpeg"
        image_array = download_blob_to_image(
            ref_blob_service_client,
            BlobStorageLocation(blobContainer=installation_code, blobName=img_path),
        )
        polygon_path = f"{tag_id}_{inspection_description}/reference_polygon.json"
        polygon_json = download_blob_to_json(
            ref_blob_service_client,
            BlobStorageLocation(blobContainer=installation_code, blobName=polygon_path),
        )
        polygon = json.loads(polygon_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode polygon JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading reference image and polygon: {e}")
        raise
    return image_array, polygon
