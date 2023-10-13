import os
from datetime import datetime
from typing import Dict

import cv2


def get_file_info(file_path: str) -> Dict[str, str | int]:
    """
    Fetches the name and size of the specified file.

    Args:
        file_path (str): The path to the file whose information is to be fetched.

    Returns:
        dict: A dictionary containing the file name and file size in bytes.
    """
    file_name = os.path.basename(file_path)  # Extracts the file name from the file path
    file_size = os.path.getsize(file_path)  # Gets the file size in bytes
    return {"file_name": file_name, "file_size": file_size}


def get_file_extension(file_path: str) -> str:
    """
    Fetches the file extension of the specified file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file extension.
    """
    return os.path.splitext(file_path)[1]


def get_file_modification_time(file_path: str) -> str:
    """
    Fetches the last modification time of the specified file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The last modification time formatted as 'YYYY-MM-DD HH:MM:SS'.
    """
    mod_time_epoch = os.path.getmtime(file_path)
    mod_time = datetime.fromtimestamp(mod_time_epoch).strftime("%Y-%m-%d %H:%M:%S")
    return mod_time


def get_image_dimensions(file_path: str) -> Dict[str, int]:
    """
    Fetches the dimensions of the specified image file using OpenCV.

    Args:
        file_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the width and height of the image.
    """
    img = cv2.imread(file_path)
    height, width = img.shape[:2]
    return {"width": width, "height": height}


def get_colors(file_path: str) -> Dict[str, tuple[int]]:
    """
    Fetches the color information of the specified image file using OpenCV.

    Args:
        file_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the mean and standard deviation of color channels.
    """
    img = cv2.imread(file_path)
    mean, std = cv2.meanStdDev(img)
    mean_color = tuple(map(int, mean.flatten()))
    std_color = tuple(map(int, std.flatten()))

    return {"mean_color": mean_color, "std_color": std_color}


def get_aspect_ratio(file_path: str) -> Dict[str, float]:
    """
    Fetches the aspect ratio of the specified image file using OpenCV.

    Args:
        file_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the aspect ratio of the image.
    """
    img = cv2.imread(file_path)
    height, width = img.shape[:2]
    aspect_ratio = width / height

    return {"aspect_ratio": aspect_ratio}
