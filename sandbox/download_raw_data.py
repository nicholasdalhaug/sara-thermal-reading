# Requires source connection string

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

directory_path = os.path.dirname(__file__)
RAW_DATA_FOLDER = f"{directory_path}/data/raw"


def download_raw_data(
    local_raw_data_folder: str, source_blob_container_connection_string: str
) -> None:
    src_blob_service_client = BlobServiceClient.from_connection_string(
        source_blob_container_connection_string
    )
    src_container_client = src_blob_service_client.get_container_client(container="nls")
    blobs_itearator = src_container_client.list_blobs()
    blob_names_fff = [
        blob.name for blob in blobs_itearator if blob.name.split(".")[-1] == "fff"
    ]

    for blob_name_fff in blob_names_fff:
        tag = blob_name_fff.split("/")[-1].split("__")[0]
        mission_folder_name = blob_name_fff.split("/")[0]

        fff_blob_client = src_container_client.get_blob_client(blob=blob_name_fff)
        fff_bytes = fff_blob_client.download_blob().readall()
        path = f"{local_raw_data_folder}/{tag}/{blob_name_fff}"
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, mode="wb") as file:
            file.write(fff_bytes)

        blobs_itearator = src_container_client.list_blobs(
            name_starts_with=f"{mission_folder_name}/{tag}"
        )
        blob_names_jpg = [
            blob.name for blob in blobs_itearator if blob.name.split(".")[-1] == "jpeg"
        ]
        if len(blob_names_jpg) == 1:
            blob_name_jpg = blob_names_jpg[0]

            jpg_blob_client = src_container_client.get_blob_client(blob=blob_name_jpg)
            jpg_bytes = jpg_blob_client.download_blob().readall()
            path = f"{local_raw_data_folder}/{tag}/{blob_name_jpg}"
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            with open(path, mode="wb") as file:
                file.write(jpg_bytes)


if __name__ == "__main__":
    load_dotenv()
    source_blob_container_connection_string = str(
        os.getenv("SOURCE_STORAGE_CONNECTION_STRING")
    )

    download_raw_data(RAW_DATA_FOLDER, source_blob_container_connection_string)
