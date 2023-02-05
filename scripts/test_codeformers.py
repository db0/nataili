import time

from PIL import Image

from nataili.codeformers import codeformers
from nataili.model_manager.codeformer import CodeFormerModelManager
from nataili.util.logger import logger

image = Image.open("01.png").convert("RGB")

mm = CodeFormerModelManager()
mm.load("CodeFormers")
upscaler = codeformers(
mm.loaded_models["CodeFormers"],
)
for iter in range(5):

    tick = time.time()
    results = upscaler(input_image=image)
    logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")
