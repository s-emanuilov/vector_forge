from enum import Enum

DEVICE = "cpu"


class Models(str, Enum):
    CLIP = "openai/clip-vit-base-patch32"
    Xception = "keras.applications.xception"
    VGG16 = "keras.application.vgg16"
    VGG19 = "keras.application.vgg19"
