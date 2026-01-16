import glob
import os
import shutil
from pathlib import Path

directory_path = os.path.dirname(__file__)
DATA_FOLDER = f"{directory_path}/data"


def create_reference_images() -> None:
    raw_tag_folder_paths = glob.glob(f"{DATA_FOLDER}/raw/*", recursive=True)

    for raw_tag_folder_path in raw_tag_folder_paths:
        tag = raw_tag_folder_path.split("/")[-1]
        fff_paths = glob.glob(f"{DATA_FOLDER}/raw/{tag}/**/*.fff", recursive=True)
        fff_path = fff_paths[0]  # Just take one, anyone
        fff_folder = "/".join(fff_path.split("/")[:-1:])
        # jpg_paths = glob.glob(f"{fff_folder}/*.jpeg")
        # assert len(jpg_paths) == 1
        # jpg_path = jpg_paths[0]

        dest_dir = f"{DATA_FOLDER}/ref/{tag}"
        dest_fff_path = f"{dest_dir}/reference_image.fff"
        Path(dest_dir).mkdir(exist_ok=True, parents=True)
        shutil.copy(fff_path, dest_fff_path)

        # dest_jpg_path = f"{dest_dir}/reference_image.jpeg"
        # shutil.copy(jpg_path, dest_jpg_path)


if __name__ == "__main__":
    create_reference_images()
