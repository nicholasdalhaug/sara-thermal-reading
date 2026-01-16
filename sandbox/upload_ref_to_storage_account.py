# Requires REFERENCE_STORAGE_CONNECTION_STRING

import glob
import os

from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

directory_path = os.path.dirname(__file__)
DATA_FOLDER = f"{directory_path}/data"


def upload_ref_data() -> None:
    tag_folders = glob.glob(f"{DATA_FOLDER}/ref/*")
    for tag_folder in tag_folders:
        tag = tag_folder.split("/")[-1]
        if len(glob.glob(f"{DATA_FOLDER}/ref/{tag}/*")) == 2:
            # upload_single_file(tag, "reference_image.jpeg")
            upload_single_file(tag, "reference_image.fff")
            upload_single_file(tag, "reference_polygon.json")


def upload_single_file(tag: str, filename: str) -> None:
    dest_blob_container_connection_string = str(
        os.getenv("REFERENCE_STORAGE_CONNECTION_STRING")
    )

    blob_service_client = BlobServiceClient.from_connection_string(
        dest_blob_container_connection_string
    )
    blob_folder_name = f"{tag}_Leakage point"
    settings: ContentSettings = ContentSettings(content_type="application/octet-stream")

    file_path = f"{DATA_FOLDER}/ref/{tag}/{filename}"
    blob_client = blob_service_client.get_blob_client(
        container="nls",
        blob=f"{blob_folder_name}/{filename}",
    )
    with open(file_path, "rb") as file:
        data = file.read()
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=settings,
    )


if __name__ == "__main__":
    load_dotenv()
    upload_ref_data()
