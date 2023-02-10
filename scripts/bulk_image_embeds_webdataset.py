import os
import webdataset as wds

from PIL import Image
from tqdm import tqdm

from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger
from nataili.cache import Cache
from nataili.clip.image import ImageEmbed
from nataili.clip.text import TextEmbed


images = []
dataset = (
wds.WebDataset("diffusiondb-{0000..0041}.tar")
.shuffle(100)
.decode("pil")
)

mm = ClipModelManager()

mm.load("ViT-H-14")

cache = Cache(mm.loaded_models["ViT-H-14"]["cache_name"], cache_parentname="embeds", cache_subname="image")

image_embed = ImageEmbed(mm.loaded_models["ViT-H-14"], cache)

db_files = cache.get_all()
db_files = {file for file in db_files}

logger.info(f"Found {len(db_files)} files in cache")

batch_size = 64

batch = []

for file in tqdm(dataset, disable=False):
    if file['__key__'] in db_files:
        continue
    image = {}
    image['pil_image'] = file['webp']
    image['filename'] = file['__key__']
    batch.append(image)
    if len(batch) == batch_size:
        image_embed._batch(batch)
        del batch
        batch = []
        



