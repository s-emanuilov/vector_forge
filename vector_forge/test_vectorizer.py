import unittest

import numpy as np

from vector_forge import vectorizer


class TestVectorizer(unittest.TestCase):
    def setUp(self):
        self.vectorizer_clip = vectorizer.Vectorizer(model=vectorizer.Models.CLIP)
        self.vectorizer_xception = vectorizer.Vectorizer(
            model=vectorizer.Models.XCEPTION
        )
        self.vectorizer_vgg16 = vectorizer.Vectorizer(model=vectorizer.Models.VGG16)
        self.sample_image_path = (
            "test_data/sample.jpg"
        )
        self.sample_text = "This is a sample text for testing."

    def test_image_to_vector(self):
        # Testing with CLIP
        vector = self.vectorizer_clip.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

        # Testing with Xception
        vector = self.vectorizer_xception.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

        # Testing with VGG16
        vector = self.vectorizer_vgg16.image_to_vector(self.sample_image_path)
        self.assertIsInstance(vector, np.ndarray)

    def test_text_to_vector(self):
        # Testing text to vector with CLIP as it's the only model supporting text for now
        vector = self.vectorizer_clip.text_to_vector(self.sample_text)
        self.assertIsInstance(vector, np.ndarray)

    def test_load_from_folder(self):
        vectors = list(self.vectorizer_clip.load_from_folder("test_data"))
        self.assertTrue(len(vectors) > 0)
        self.assertIsInstance(vectors[0], np.ndarray)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
