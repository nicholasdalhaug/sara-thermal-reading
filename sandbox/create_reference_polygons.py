import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_reference_polygon,
)
from sara_thermal_reading.file_io.fff_loader import load_fff

directory_path = os.path.dirname(__file__)
DATA_FOLDER = f"{directory_path}/data"


def create_reference_polygons() -> None:
    reference_image_paths = glob.glob(f"{DATA_FOLDER}/ref/**/*.fff", recursive=True)

    for reference_image_path in reference_image_paths:
        reference_image_folder = "/".join(
            reference_image_path.split("/")[:-1:]
        )  # Remove file from path
        json_path = f"{reference_image_folder}/reference_polygon.json"
        # jpg_path = f"{reference_image_folder}/reference_image.jpeg"

        plt.ion()

        # plt.figure("JPG")
        # plt.clf()
        # image = plt.imread(jpg_path)
        # plt.imshow(image)
        # plt.axis("off")

        fff_image = load_fff(reference_image_path)
        fff_image = np.clip(fff_image, -10, 20)
        plt.figure("Thermal")
        plt.clf()
        plt.imshow(fff_image)
        plt.colorbar()
        plt.axis("off")

        plt.show()
        plt.pause(0.1)

        create_reference_polygon(Path(reference_image_path), Path(json_path))


if __name__ == "__main__":
    create_reference_polygons()
