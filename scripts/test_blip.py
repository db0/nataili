from PIL import Image

from nataili.model_manager.blip import BlipModelManager
from nataili.blip import Caption
from nataili.util.logger import logger

image = Image.open("test.png")

mm = BlipModelManager()

mm.load("BLIP")

blip = Caption(mm.loaded_models["BLIP"])

logger.generation(f"caption: {blip(image)}")
