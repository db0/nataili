from PIL import Image

from nataili.clip import CoCa
from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger

image = Image.open("01.png").convert("RGB")

mm = ClipModelManager(download_reference=False)

mm.load("coca_ViT-L-14")

coca = CoCa(
    mm.loaded_models["coca_ViT-L-14"]["model"],
    mm.loaded_models["coca_ViT-L-14"]["transform"],
    mm.loaded_models["coca_ViT-L-14"]["device"],
    mm.loaded_models["coca_ViT-L-14"]["half_precision"],
)

logger.generation(coca(image))
