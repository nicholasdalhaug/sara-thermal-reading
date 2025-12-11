import os
from io import BytesIO
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BlobStorageLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    blob_container: str = Field(..., alias="blobContainer")
    blob_name: str = Field(..., alias="blobName")

    @field_validator("blob_container")
    def validate_blob_container(cls, v: str) -> str:
        if not v:
            raise ValueError("blobContainer cannot be empty")
        return v

    @field_validator("blob_name")
    def validate_blob_name(cls, v: str) -> str:
        if not v:
            raise ValueError("blobName cannot be empty")
        return v


def download_blob_to_bytes(
    blob_service_client: BlobServiceClient, blob_storage_location: BlobStorageLocation
) -> bytes:
    blob_client = blob_service_client.get_blob_client(
        container=blob_storage_location.blob_container,
        blob=blob_storage_location.blob_name,
    )
    return blob_client.download_blob().readall()


def download_blob_to_image(
    blob_service_client: BlobServiceClient, blob_storage_location: BlobStorageLocation
) -> NDArray[np.uint8]:
    blob_data = download_blob_to_bytes(blob_service_client, blob_storage_location)
    image = Image.open(BytesIO(blob_data))
    return np.array(image)


def download_blob_to_json(
    blob_service_client: BlobServiceClient,
    blob_storage_location: BlobStorageLocation,
) -> str:
    blob_data = download_blob_to_bytes(blob_service_client, blob_storage_location)
    return blob_data.decode("utf-8")


def upload_bytes_to_blob(
    blob_service_client: BlobServiceClient,
    blob_storage_location: BlobStorageLocation,
    data: BytesIO,
    content_type: str = "application/octet-stream",
) -> None:
    try:
        blob_client = blob_service_client.get_blob_client(
            container=blob_storage_location.blob_container,
            blob=blob_storage_location.blob_name,
        )
        settings: ContentSettings = ContentSettings(content_type=content_type)  # type: ignore

        blob_data = data.getvalue()
        buffer_size = len(blob_data)
        logger.debug(
            f"Uploading {buffer_size} bytes to blob {blob_storage_location.blob_name}"
        )

        blob_client.upload_blob(
            blob_data,
            overwrite=True,
            content_settings=settings,
        )

        logger.debug(f"Successfully uploaded blob {blob_storage_location.blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload blob {blob_storage_location.blob_name}: {e}")
        raise


def upload_image_to_blob(
    blob_service_client: BlobServiceClient,
    blob_storage_location: "BlobStorageLocation",
    image: NDArray[np.uint8],
) -> None:

    def ndarray_to_bytesio(arr: np.ndarray, format: str = "JPEG") -> BytesIO:
        img = Image.fromarray(arr)

        if img.mode == "RGBA" and format.upper() == "JPEG":
            img = img.convert("RGB")

        buf = BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return buf

    logger.debug(f"Uploading image to {blob_storage_location.blob_name}")

    upload_bytes_to_blob(
        blob_service_client,
        blob_storage_location,
        ndarray_to_bytesio(image, format="JPEG"),
        content_type="image/jpeg",
    )
