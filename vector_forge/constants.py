from enum import Enum

DEVICE = "cpu"


class Models(str, Enum):
    CLIP_B_P32 = "openai/clip-vit-base-patch32"
    CLIP_L_P14 = "openai/clip-vit-large-patch14"
    Xception = "keras.applications.xception"
    VGG16 = "keras.application.vgg16"
    VGG19 = "keras.application.vgg19"
