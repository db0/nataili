import os

from PIL import Image

from nataili.model_manager.clip import ClipModelManager
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger


images = []

directory = "test_images"

mm = ClipModelManager()

mm.load("ViT-L/14")

interrogator = Interrogator(
    mm.loaded_models["ViT-L/14"],
)

for file in os.listdir(directory):
    image  = Image.open(f"{directory}/{file}").convert("RGB")
    results = interrogator(image=image, text_array=None, rank=True, top_count=5)
    """
    or
    results = interrogator(filename=file, directory=directory, text_array=None, rank=True, top_count=5)
    """
    logger.generation(results)
