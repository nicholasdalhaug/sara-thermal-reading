import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from sara_thermal_reading.file_io.fff_loader import load_fff
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)

directory_path = os.path.dirname(__file__)
DATA_FOLDER = f"{directory_path}/data"


def run_matching_single(tag: str) -> None:
    thermal_image_paths = glob.glob(f"{DATA_FOLDER}/raw/{tag}/**/*.fff")
    ref_image_path = f"{DATA_FOLDER}/ref/{tag}/reference_image.fff"
    polygon_path = f"{DATA_FOLDER}/ref/{tag}/reference_polygon.json"

    for thermal_image_path in thermal_image_paths:
        ref_image = load_fff(ref_image_path)
        thermal_image = load_fff(thermal_image_path)
        with open(polygon_path, "r") as file:
            polygon_list = json.load(file)
        polygon_np = np.array(polygon_list)

        a_min = max(np.min(ref_image), np.min(thermal_image))
        a_max = min(np.max(ref_image), np.max(thermal_image))
        ref_uint8 = (
            (np.clip(ref_image, a_min=a_min, a_max=a_max) - a_min)
            / (a_max - a_min)
            * 255
        ).astype("uint8")
        thermal_uint8 = (
            (np.clip(thermal_image, a_min=a_min, a_max=a_max) - a_min)
            / (a_max - a_min)
            * 255
        ).astype("uint8")

        warped_polygon, aligned_image = align_two_images_orb_bf_cv2(
            ref_uint8,
            thermal_uint8,
            polygon_list,
        )
        warped_polygon_np = np.array(warped_polygon)

        plt.ioff()

        plt.figure("Comparison")
        fig = plt.gcf()
        fig.set_figwidth(13)
        fig.set_figheight(13)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.title("Reference image")
        # plt.imshow(ref_uint8)
        plt.imshow(ref_image, vmin=0, vmax=50)
        plt.fill(
            polygon_np[:, 0],
            polygon_np[:, 1],
            facecolor="red",
            edgecolor="white",
            linewidth=2,
            alpha=0.5,
        )
        plt.axis("off")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.title("Thermal image")
        # plt.imshow(thermal_uint8)
        plt.imshow(thermal_image, vmin=0, vmax=50)
        plt.fill(
            warped_polygon_np[:, 0],
            warped_polygon_np[:, 1],
            facecolor="red",
            edgecolor="white",
            linewidth=2,
            alpha=0.5,
        )
        plt.axis("off")
        plt.colorbar()

        plt.show()


if __name__ == "__main__":
    # tag = "52-LQ-4005"
    # tag = "52-LQ-4006"
    # tag = "52-LQ-4010"
    # tag = "52-LQ-5248"
    tag = "52-LQ-5254"
    # tag = "52-LQ-5324"
    run_matching_single(tag)
