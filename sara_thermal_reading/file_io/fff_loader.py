import numpy as np
from flirpy.io.fff import Fff


def load_fff(file_path: str) -> np.ndarray:
    """
    Load an FFF file and return the temperature (Celsius) image as a numpy array.
    """
    fff = Fff(file_path)
    image = fff.get_radiometric_image(dtype="float")
    return image


def load_fff_from_bytes(data: bytes) -> np.ndarray:
    """
    Load FFF data from bytes and return the temperature (Celsius) image as a numpy array.
    """
    fff = Fff(data)
    image = fff.get_radiometric_image(dtype="float")
    return image
