import os

import PIL

from nataili import Interrogator, ModelManager, logger

images = []

directory = "test_images"

for file in os.listdir(directory):
    pil_image = PIL.Image.open(f"{directory}/{file}").convert("RGB")
    images.append({"pil_image": pil_image, "filename": file})

mm = ModelManager()

mm.clip.load("ViT-L/14")

interrogator = Interrogator(
    mm.clip.loaded_models["ViT-L/14"],
)

for image in images:
    results = interrogator(image['pil_image'], text_array=None, rank=True, top_count=5)
    logger.generation(results)
