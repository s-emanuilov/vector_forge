import os
from typing import Callable

import cv2
import numpy as np
import torch
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.applications.vgg19 import (
    preprocess_input as vgg19_preprocess_input,
)
from tensorflow.keras.applications.xception import (
    preprocess_input as xcpetion_preprocess_input,
)
from tensorflow.keras.preprocessing import image

from .constants import Models

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
        model: Models = Models.CLIP_B_P32,
        image_preprocessor: Callable[[np.ndarray], np.ndarray] = None,
        normalization: bool = False,
    ):
        """
        Initializes the Vectorizer with the specified model and an optional image preprocessor function.

        Args:
            model (Models, optional): The model to be used for vectorization. Defaults to Models.CLIP_B_P32.
            image_preprocessor (callable, optional): A function to preprocess the image before resizing.
                                                     This function should accept a single argument, the image,
                                                     and return the preprocessed image. Defaults to None.
            normalization (bool, optional): Whether to normalize the generated vectors. Defaults to False.
        """
        self.model = model
        self.image_preprocessor = image_preprocessor
        self.normalization = normalization
        if model == Models.CLIP_B_P32 or model == Models.CLIP_L_P14:
            from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel

            self.processor = CLIPProcessor.from_pretrained(model.value)
            self.model_instance = CLIPModel.from_pretrained(model.value).to(DEVICE)
        elif model == Models.CLIP_B_P32_OV or model == Models.CLIP_L_P14_OV:
            from huggingface_hub import snapshot_download
            from transformers import CLIPProcessor
            from openvino.runtime import Core

            ov_path = snapshot_download(repo_id=model.value)
            self.processor = CLIPProcessor.from_pretrained(model.value)
            model_file = (
                "clip-vit-base-patch32.xml"
                if model == Models.CLIP_B_P32_OV
                else "clip-vit-large-patch14.xml"
            )
            ov_model_xml = os.path.join(ov_path, model_file)
            core = Core()
            ov_model = core.read_model(model=ov_model_xml)
            # Compile model for loading on device
            self.model_instance = core.compile_model(ov_model)
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
        input_image: str,
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
        if self.model in (Models.CLIP_B_P32, Models.CLIP_L_P14):
            input_image = self._prepare_image_clip(input_image)
            image_tensor = self.processor(
                text=None, images=input_image, return_tensors="pt"
            )["pixel_values"]
            img_emb = self.model_instance.get_image_features(image_tensor)
        elif self.model in (Models.CLIP_B_P32_OV, Models.CLIP_L_P14_OV):
            input_image = self._prepare_image_clip(input_image)
            image_tensor = self.processor(
                text=["", ""], images=[input_image], return_tensors="pt"
            )
            img_emb = self.model_instance(dict(image_tensor))["image_embeds"]
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
        if self.model in (Models.CLIP_B_P32, Models.CLIP_L_P14):
            from transformers import CLIPTokenizerFast

            tokenizer = CLIPTokenizerFast.from_pretrained(self.model.value)
            inputs = tokenizer(input_text, return_tensors="pt")
            text_emb = self.model_instance.get_text_features(**inputs)
            result = text_emb[0].cpu().detach().numpy()
        elif self.model in (Models.CLIP_B_P32_OV, Models.CLIP_L_P14_OV):
            # Generate a random "image" to pass throught layers of the OpenVino model
            # then get only text_embeds
            rand_image = np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8)
            text_tensor = self.processor(
                text=[input_text], images=[rand_image], return_tensors="pt"
            )
            result = self.model_instance(dict(text_tensor))["text_embeds"][0]
        else:
            raise ValueError(f"Unsupported model for text: {self.model}")

        if self.normalization:
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

    @staticmethod
    def _check_image(input_image: str | np.ndarray) -> np.ndarray:
        """
        Verify if the input image exist and it is valid

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
        return input_image

    def _apply_preprocessors(self, input_image: str | np.ndarray) -> np.ndarray:
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
        input_image = self._check_image(input_image)
        input_image = self._apply_preprocessors(input_image)
        return input_image

    def _prepare_image_xception(self, input_image: str) -> np.ndarray:
        """
        Prepares the input image for vectorization using the Xception model
        by resizing and preprocessing it.

        Args:
            input_image (str): Path to the image.

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with Xception.
        """
        input_image = image.load_img(input_image, target_size=(299, 299))
        input_image = image.img_to_array(input_image)
        input_image = self._apply_preprocessors(input_image)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = xcpetion_preprocess_input(input_image)
        return input_image

    def _prepare_image_vgg16(self, input_image: str) -> np.ndarray:
        """
        Prepares the input image for vectorization using the VGG16 model
        by resizing and preprocessing it.

        Args:
            input_image (str): Path to the image

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with VGG16.
        """
        input_image = image.load_img(input_image, target_size=(224, 224))
        input_image = image.img_to_array(input_image)
        input_image = self._apply_preprocessors(input_image)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = vgg16_preprocess_input(input_image)
        return input_image

    def _prepare_image_vgg19(self, input_image: str) -> np.ndarray:
        """
        Prepares the input image for vectorization using the VGG19 model
        by resizing and preprocessing it.

        Args:
            input_image (str): Path to the image

        Returns:
            np.ndarray: The preprocessed image ready for vectorization with VGG19.
        """
        input_image = image.load_img(input_image, target_size=(224, 224))
        input_image = image.img_to_array(input_image)
        input_image = self._apply_preprocessors(input_image)
        input_image = np.expand_dims(input_image, axis=0)
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
            result[0].cpu().detach().numpy()
            if self.model in (Models.CLIP_B_P32, Models.CLIP_L_P14)
            else result[0]
        )

        if self.normalization:
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
        folder,
        batch_size=32,
        return_type="numpy",
        file_info_extractor=None,
        save_to_index=None,
    ):
        """
        Process images in batches from a folder.

        Args:
            folder (str): Directory containing images.
            batch_size (int): Number of images to process in each batch.
            return_type (str): Format for returned vectors.
            file_info_extractor (callable): Function to extract file information.
            save_to_index (str): Path to save the index of processed files.

        Yields:
            Batch results as specified by return_type.
        """
        image_batch = []
        index = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Read image, skip if not valid
                img = cv2.imread(file_path)
                if img is not None:
                    image_batch.append(file_path)
                    index.append(file_path)

                    if len(image_batch) == batch_size:
                        yield from self.process_batch(
                            image_batch, return_type, file_info_extractor
                        )
                        image_batch = []

        # Process remaining images in the last batch
        if image_batch:
            yield from self.process_batch(image_batch, return_type, file_info_extractor)

        if save_to_index:
            with open(save_to_index, "w") as f:
                for path in index:
                    f.write(f"{path}\n")

    def process_batch(self, image_batch, return_type, file_info_extractor):
        """
        Process a batch of images for vectorization.

        Args:
            image_batch (list): List of image file paths.
            return_type (str): Format for returned vectors.
            file_info_extractor (callable): Function to extract file information.

        Returns:
            List of processed vectors or tuples of vectors and file info.
        """
        vectors = []
        for image_path in image_batch:
            vector = self.image_to_vector(image_path, return_type)
            if file_info_extractor:
                file_info = file_info_extractor(image_path)
                vectors.append((vector, file_info))
            else:
                vectors.append(vector)

        return vectors
