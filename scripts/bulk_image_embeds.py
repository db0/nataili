import os

import PIL
from tqdm import tqdm

from nataili import ModelManager, logger
from nataili.clip import ImageEmbed
from nataili.cache import Cache



mm = ModelManager()

mm.clip.load("ViT-H-14")

cache = Cache(mm.clip.loaded_models["ViT-H-14"]["cache_name"], cache_parentname="embeds", cache_subname="image")

image_embed = ImageEmbed(mm.clip.loaded_models["ViT-H-14"], cache)

images = []

directory = "Y:/diffusiondb"

batch_size = 256

for file in os.listdir(directory):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".webp"):
        images.append({"filename": file})

for i in tqdm(range(0, len(images), batch_size)):
    batch = images[i:i+batch_size]
    for image in batch:
        image['pil_image'] = PIL.Image.open(f"{directory}/{image['filename']}").convert("RGB")
    image_embed = ImageEmbed(mm.clip.loaded_models["ViT-H-14"], cache)
    image_embed.batch(batch, batch_size=batch_size)
    cache.flush()
    del batch
        



