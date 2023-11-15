import unittest

import numpy as np
from keras.applications import VGG19, vgg19
from keras.applications import Xception, VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.preprocessing import image

from vector_forge import Vectorizer, Models
from vector_forge import image_preprocessors, info_extractors


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


class TestVectorizer(unittest.TestCase):
    def setUp(self):
        self.vectorizer_clip = Vectorizer(model=Models.CLIP_B_P32)
        self.vectorizer_clip_preprocessor = Vectorizer(
            model=Models.CLIP_B_P32,
            image_preprocessor=image_preprocessors.threshold_image,
        )
        self.vectorizer_xception = Vectorizer(model=Models.Xception)
        self.vectorizer_vgg16 = Vectorizer(model=Models.VGG16)
        self.vectorizer_vgg19 = Vectorizer(model=Models.VGG19)
        self.sample_image_path = "test_data/sample.jpg"
        self.sample_text = "This is a sample text for testing."

    def test_image_to_vector(self):
        # Testing with CLIP_B_P32
        vector = self.vectorizer_clip.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

        # Testing with CLIP_B_P32 and preprocessor
        vector = self.vectorizer_clip_preprocessor.image_to_vector(
            self.sample_image_path
        )
        self.assertIsInstance(vector, np.ndarray)

        # Testing with Xception
        vector = self.vectorizer_xception.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

        # Testing with VGG16
        vector = self.vectorizer_vgg16.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

        # Testing with VGG19
        vector = self.vectorizer_vgg19.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

    def test_text_to_vector(self):
        # Testing text to vector with CLIP_B_P32 as it's the only model supporting text for now
        vector = self.vectorizer_clip.text_to_vector(self.sample_text)
        self.assertIsInstance(vector, np.ndarray)

    def test_similarity(self):
        vectorizer = Vectorizer(model=Models.CLIP_B_P32_OV)

        # Generate text embeddings
        text_embedding_1 = vectorizer.text_to_vector("The dog is on the couch")
        text_embedding_2 = vectorizer.text_to_vector("The dog just sat on the couch")

        # Calculate similarity
        similarity = cosine_similarity(text_embedding_1, text_embedding_2)

        # Test if similarity is above 96%
        self.assertGreaterEqual(similarity, 0.96, "Similarity is below 96%")

    def test_load_from_folder(self):
        vectors = list(
            self.vectorizer_clip.load_from_folder(
                "test_data", file_info_extractor=info_extractors.get_file_info
            )
        )
        self.assertTrue(len(vectors) > 0)
        self.assertIsInstance(vectors[0][0], np.ndarray)

    def test_xception_vector(self):
        # Standalone Xception processing
        base_model = Xception(weights="imagenet", include_top=False, pooling="avg")
        img_path = "test_data/birds.jpg"
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception_preprocess(x)
        expected_vector = base_model.predict(x).flatten()

        # Vectorizer Xception processing
        vectorizer = Vectorizer(model=Models.Xception)
        vector = vectorizer.image_to_vector(img_path)

        np.testing.assert_array_almost_equal(vector, expected_vector, decimal=6)

    def test_vgg16_vector(self):
        # Standalone VGG16 processing
        base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        img_path = "test_data/birds.jpg"
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg16_preprocess(x)
        expected_vector = base_model.predict(x).flatten()

        # Vectorizer VGG16 processing
        vectorizer = Vectorizer(model=Models.VGG16)
        vector = vectorizer.image_to_vector(img_path)

        np.testing.assert_array_almost_equal(vector, expected_vector, decimal=6)

    def test_vgg19_vector(self):
        # Standalone VGG19 processing
        base_model = VGG19(weights="imagenet", include_top=False, pooling="avg")
        img_path = "test_data/birds.jpg"
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg19.preprocess_input(x)
        expected_vector = base_model.predict(x).flatten()

        # Vectorizer VGG19 processing
        vectorizer = Vectorizer(model=Models.VGG19)
        vector = vectorizer.image_to_vector(img_path)

        np.testing.assert_array_almost_equal(vector, expected_vector, decimal=6)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
