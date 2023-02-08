import os

from PIL import Image
from tqdm import tqdm

from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger
from nataili.cache import Cache
from nataili.clip.image import ImageEmbed
from nataili.clip.text import TextEmbed


images = []

directory = "test_images"

mm = ClipModelManager()

mm.load("ViT-H-14")

cache = Cache(mm.loaded_models["ViT-H-14"]["cache_name"], cache_parentname="embeds", cache_subname="image")

image_embed = ImageEmbed(mm.loaded_models["ViT-H-14"], cache)

images = []

directory = "Y:/diffusiondb"

file_list = [os.path.splitext(file)[0] for file in os.listdir(directory) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".webp")]

logger.info(f"Found {len(file_list)} images in directory")

filtered = cache.filter_list(file_list)

filtered = {key: None for key in filtered}

logger.info(f"Found {len(filtered)} images in cache")

batch_size = 256

for file in tqdm(os.listdir(directory)):
    if os.path.splitext(file)[0] in filtered:
        images.append({"filename": file})

logger.info(f"Found {len(images)} images to process")

for i in tqdm(range(0, len(images), batch_size)):
    batch = images[i:i+batch_size]
    for image in batch:
        image['pil_image'] = Image.open(f"{directory}/{image['filename']}").convert("RGB")
    image_embed._batch(batch)
    del batch
        



