import os
from typing import Callable

import cv2
import numpy as np
import torch

from .constants import Models
from .image_preprocessors import convert_to_rgb_expand

# Check if GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vectorizer:
    """
    Vectorizer class for converting images and text into vector representations
    using specified models like CLIP, Xception, and VGG16.

    Attributes:
        model (Models): The model to be used for vectorization.
        model_instance (Model): The instance of the specified model.
        processor (CLIPProcessor, optional): Processor for CLIP model, if applicable.
    """

    def __init__(
        self,
        model: Models = Models.CLIP,
        image_preprocessor: Callable[[np.ndarray], np.ndarray] = None,
    ):
        """
        Initializes the Vectorizer with the specified model and an optional image preprocessor function.

        Args:
            model (Models, optional): The model to be used for vectorization. Defaults to Models.CLIP.
            image_preprocessor (callable, optional): A function to preprocess the image before resizing.
                                                     This function should accept a single argument, the image,
                                                     and return the preprocessed image. Defaults to None.
        """
        self.model = model
        self.image_preprocessor = image_preprocessor
        if model == Models.CLIP:
            from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel

            self.processor = CLIPProcessor.from_pretrained(model.value)
            self.model_instance = CLIPModel.from_pretrained(model.value).to(DEVICE)
        elif model == Models.Xception:
            from tensorflow.keras.applications import Xception  # lazy loading

            self.model_instance = Xception(
                weights="imagenet", include_top=False, pooling="avg"
            )
        elif model == Models.VGG16:
            from tensorflow.keras.applications import VGG16  # lazy loading

            self.model_instance = VGG16(
                weights="imagenet", include_top=False, pooling="avg"
            )
        elif model == Models.VGG19:
            from tensorflow.keras.applications import VGG19  # lazy loading

            self.model_instance = VGG19(
                weights="imagenet", include_top=False, pooling="avg"
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

    def image_to_vector(
        self,
        input_image: str | np.ndarray,
        return_type: str = "numpy",
    ) -> np.ndarray | str | list:
        """
        Converts an image to a vector representation using the specified model.

        Args:
            input_image (str | np.ndarray): Path to the image or a NumPy array of the image.
            return_type (str, optional): The format in which to return the vector.
                                          Options are "numpy", "str", "list", or "2darray".
                                          Defaults to "numpy".

        Returns:
            np.ndarray | str | list: The vector representation of the image.
        """
        if self.model == Models.CLIP:
            input_image = self._prepare_image_clip(input_image)
            image_tensor = self.processor(
                text=None, images=input_image, return_tensors="pt"
            )["pixel_values"]
            img_emb = self.model_instance.get_image_features(image_tensor)
        elif self.model == Models.Xception:
            input_image = self._prepare_image_xception(input_image)
            img_emb = self.model_instance.predict(input_image, verbose=0)
        elif self.model == Models.VGG16:
            input_image = self._prepare_image_vgg16(input_image)
            img_emb = self.model_instance.predict(input_image, verbose=0)
        elif self.model == Models.VGG19:
            input_image = self._prepare_image_vgg19(input_image)
            img_emb = self.model_instance.predict(input_image, verbose=0)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        result = self._process_result(img_emb, return_type)
        return result

    def text_to_vector(
        self, input_text: str, return_type: str = "numpy"
    ) -> np.ndarray | str | list:
        """
        Converts a given text to a vector representation using the specified model.

        Args:
            input_text (str): The text to be vectorized.
            return_type (str, optional): The format in which to return the vector.
                                          Options are "numpy", "str", "list", or "2darray".
                                          Defaults to "numpy".

        Returns:
            np.ndarray | str | list: The vector representation of the text.
        """
        if self.model == Models.CLIP:
            from transformers import CLIPTokenizerFast

            tokenizer = CLIPTokenizerFast.from_pretrained(self.model.value)
            inputs = tokenizer(input_text, return_tensors="pt")
            text_emb = self.model_instance.get_text_features(**inputs)
            result = text_emb[0].cpu().detach().numpy()
        else:
            raise ValueError(f"Unsupported model for text: {self.model}")

        result /= np.linalg.norm(result, axis=0)
        if return_type == "numpy":
            return result
        elif return_type == "2darray":
            return result[np.newaxis, :]
        elif return_type == "list":
            return result.tolist()
        elif return_type == "str":
            return np.array2string(result, separator=", ").replace("\n", "")
        else:
            raise ValueError(f"Unsupported return type: {return_type}")

    def _prepare_image(self, input_image: str | np.ndarray) -> np.ndarray:
        """
        Prepares the input image for vectorization by applying image_preprocessor and check if the provided path is valid image

        Args:
            input_image (str | np.ndarray): Path to the image or a NumPy array of the image.

        Returns:
            np.ndarray: The resized image and potentially preprocessed image.
        """
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"File {input_image} does not exist!")
            input_image = cv2.imread(input_image)
            if input_image is None:
                raise ValueError(
                    f"File {input_image} is not a valid image file or the path is incorrect."
                )
        # Apply the global image preprocessor function if provided
        if self.image_preprocessor is not None:
            input_image = self.image_preprocessor(input_image)
        return input_image

    def _prepare_image_clip(self, input_image: str) -> np.ndarray:
        """
        Prepares an image for vectorization using the CLIP model by resizing it.

        This method is a wrapper around the _prepare_image method,
        ensuring consistent preprocessing for the CLIP model.

        Args:
            input_image (str): The path to the image to be preprocessed.

        Returns:
            np.ndarray: The preprocessed image as a NumPy array.
        """
        input_image = self._prepare_image(input_image)
        return input_image

    def _prepare_image_xception(self, input_image: str | np.ndarray) -> np.ndarray:
        """
        Prepares the input image for vectorization using the Xception model
        by resizing and preprocessing it.

        Args:
            input_image (str | np.ndarray): Path to the image or a NumPy array of the image.

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with Xception.
        """
        from tensorflow.keras.applications.xception import (
            preprocess_input,
        )  # Lazy import

        input_image = self._prepare_image(input_image)
        input_image = convert_to_rgb_expand(input_image)
        input_image = preprocess_input(
            input_image
        )  # This is where preprocess_input is used
        return input_image

    def _prepare_image_vgg16(self, input_image: str | np.ndarray) -> np.ndarray:
        """
        Prepares the input image for vectorization using the VGG16 model
        by resizing and preprocessing it.

        Args:
            input_image (str | np.ndarray): Path to the image or a NumPy array of the image.

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with VGG16.
        """
        from tensorflow.keras.applications.vgg16 import (
            preprocess_input as vgg16_preprocess_input,
        )

        input_image = self._prepare_image(input_image)
        input_image = convert_to_rgb_expand(input_image)
        input_image = vgg16_preprocess_input(input_image)
        return input_image

    def _prepare_image_vgg19(self, input_image: str | np.ndarray) -> np.ndarray:
        """
        Prepares the input image for vectorization using the VGG19 model
        by resizing and preprocessing it.

        Args:
            input_image (str | np.ndarray): Path to the image or a NumPy array of the image.

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with VGG19.
        """
        from tensorflow.keras.applications.vgg19 import (
            preprocess_input as vgg19_preprocess_input,
        )

        input_image = self._prepare_image(input_image)
        input_image = convert_to_rgb_expand(input_image)
        input_image = vgg19_preprocess_input(input_image)
        return input_image

    def _process_result(self, result, return_type: str) -> np.ndarray | str | list:
        """
        Processes the result vector based on the specified return type.

        Args:
            result (np.ndarray): The vector to be processed.
            return_type (str): The format in which to return the vector.
                               Options are "numpy", "str", "list", or "2darray".

        Returns:
            np.ndarray | str | list: The processed vector.
        """
        result = (
            result[0].cpu().detach().numpy() if self.model == Models.CLIP else result[0]
        )
        result /= np.linalg.norm(result, axis=0)
        if return_type == "str":
            return np.array2string(result, separator=", ")
        elif return_type == "list":
            return result.tolist()
        elif return_type == "numpy":
            return result
        elif return_type == "2darray":
            return result[np.newaxis, :]
        else:
            raise ValueError(f"Unsupported return type: {return_type}")

    def load_from_folder(
        self,
        folder: str,
        return_type: str = "numpy",
        save_to_index: str = None,
        file_info_extractor: callable = None,
    ):
        """
        Loads images from a specified folder, converts them to vectors,
        and yields the vectors one by one. Optionally executes a custom
        function on each file and yields the result alongside the vector.

        Args:
            folder (str): Path to the folder containing images.
            return_type (str, optional): The format in which to return the vectors.
                                          Options are "numpy", "str", "list", or "2darray".
                                          Defaults to "numpy".
            save_to_index (str): The name of the file to save an index of files processed.
                                 If None, the index is not saved. Default is None.
            file_info_extractor (callable, optional): A function to execute on each file.
                                                            The function should accept a file path as argument
                                                            and will be executed if provided.
                                                            Defaults to None.

        Yields:
            np.ndarray | str | list: The vector representation of each image if
                                      file_processing_function is None.
            or
            tuple: A tuple containing the vector representation of each image
                   and the result of file_processing_function if it is provided.
        """
        index = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Attempt to read the file as an image
                img = cv2.imread(file_path)
                if img is not None:
                    vector = self.image_to_vector(file_path, return_type=return_type)
                    index.append(file_path)

                    # Execute the specified function on the file, if provided
                    if file_info_extractor is not None:
                        file_info_result = file_info_extractor(file_path)
                        yield vector, file_info_result
                    else:
                        yield vector

        if save_to_index:
            with open(save_to_index, "w") as f:
                # Write each file path to a new line in the index file
                for path in index:
                    f.write("%s\n" % path)
