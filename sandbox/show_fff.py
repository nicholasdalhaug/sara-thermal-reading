import os

import matplotlib.pyplot as plt

from sara_thermal_reading.file_io.fff_loader import load_fff

directory_path = os.path.dirname(__file__)
DATA_FOLDER = f"{directory_path}/data"
thermal_image_path = f"{DATA_FOLDER}/xxx.fff"


def show_thermal_image() -> None:
    thermal_image = load_fff(thermal_image_path)

    plt.ioff()

    plt.figure("Thermal image")
    plt.imshow(thermal_image)
    plt.axis("off")
    plt.colorbar()

    plt.show()


def change_metadata() -> None:
    with open(thermal_image_path, "rb") as file:
        data = file.read()

    camera_model = b""
    camera_model_padded = camera_model.ljust(32, b"\x00")
    assert len(camera_model_padded) == 32
    camera_part_number = b""
    camera_part_number_padded = camera_part_number.ljust(16, b"\x00")
    assert len(camera_part_number_padded) == 16
    camera_serial_number = b""
    camera_serial_number_padded = camera_serial_number.ljust(16, b"\x00")
    assert len(camera_serial_number_padded) == 16
    camera_software = b""
    camera_software_padded = camera_software.ljust(16, b"\x00")
    assert len(camera_software_padded) == 16

    lens_model = b""
    lens_model_padded = lens_model.ljust(32, b"\x00")
    assert len(lens_model_padded) == 32
    lens_part_number = b""
    lens_part_number_padded = lens_part_number.ljust(16, b"\x00")
    assert len(lens_part_number_padded) == 16
    lens_serial_number = b""
    lens_serial_number_padded = lens_serial_number.ljust(16, b"\x00")
    assert len(lens_serial_number_padded) == 16

    data_arr = bytearray(data)

    place = 323700
    data_arr[place : place + 32] = camera_model_padded
    place += 32
    data_arr[place : place + 16] = camera_part_number_padded
    place += 16
    data_arr[place : place + 16] = camera_serial_number_padded
    place += 16
    data_arr[place : place + 16] = camera_software_padded

    place = 323856
    data_arr[place : place + 32] = lens_model_padded
    place += 32
    data_arr[place : place + 16] = lens_part_number_padded
    place += 16
    data_arr[place : place + 16] = lens_serial_number_padded

    data_changed = bytes(data_arr)

    filename_new_arr = thermal_image_path.split(".")
    filename_new_arr[0] += "_changed"
    filename_new = ".".join(filename_new_arr)

    with open(filename_new, "wb") as file:
        file.write(data_changed)
    pass


if __name__ == "__main__":
    show_thermal_image()
    # change_metadata()
