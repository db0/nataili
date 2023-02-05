import time

from PIL import Image

from nataili.esrgan import esrgan
from nataili.model_manager.esrgan import EsrganModelManager
from nataili.util.logger import logger

image = Image.open("01.png").convert("RGB")

mm = EsrganModelManager()

mm.load("RealESRGAN_x4plus")

upscaler = esrgan(mm.loaded_models["RealESRGAN_x4plus"])

for i in range(10):
    tick = time.time()
    results = upscaler(input_image=image)
    logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")
