import time

from PIL import Image

from nataili.gfpgan import gfpgan
from nataili.model_manager.gfpgan import GfpganModelManager
from nataili.util.logger import logger

image = Image.open("01.png").convert("RGB")

mm = GfpganModelManager()

mm.load("GFPGAN")

facefixer = gfpgan(mm.loaded_models["GFPGAN"])

for i in range(10):
    tick = time.time()
    results = facefixer(input_image=image, strength=1.0)
    logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")
