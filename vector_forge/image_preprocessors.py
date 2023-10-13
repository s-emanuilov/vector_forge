import cv2
import numpy as np


def resize_image(image: np.ndarray, width: int) -> np.ndarray:
    """
    Resizes the given image to the specified width while maintaining the aspect ratio.

    Args:
        image (np.ndarray): The image to be resized.
        width (int): The desired width of the output image.

    Returns:
        np.ndarray: The resized image.
    """
    r = width / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    return cv2.resize(
        image,
        dim,
        interpolation=cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LANCZOS4,
    )


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts the given image to grayscale.

    Args:
        image (np.ndarray): The image to be converted.

    Returns:
        np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the given image to have pixel values between 0 and 1.

    Args:
        image (np.ndarray): The image to be normalized.

    Returns:
        np.ndarray: The normalized image.
    """
    return image / 255.0


def threshold_image(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    """
    Applies a binary threshold to the given image.

    Args:
        image (np.ndarray): The image to be thresholded.
        threshold_value (int, optional): The threshold value. Defaults to 127.

    Returns:
        np.ndarray: The thresholded image.
    """
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts the given image from BGR to RGB color space.

    Args:
        image (np.ndarray): The image to be converted.

    Returns:
        np.ndarray: The converted image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_rgb_expand(image: np.ndarray) -> np.ndarray:
    """
    Converts the given image from BGR to RGB color space and expands its dimensions.

    The function first changes the color space of the image from BGR to RGB using OpenCV's cvtColor function.
    Then, it expands the dimensions of the image by adding an extra dimension at the start.
    This is often needed as input preparation for certain machine learning models which expect a batch of images.

    Args:
        image (np.ndarray): The input image in BGR color space as a NumPy array.

    Returns:
        np.ndarray: The image in RGB color space with an expanded dimension, as a NumPy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    return image
