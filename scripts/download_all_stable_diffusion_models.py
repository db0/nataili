from nataili.model_manager.compvis import CompVisModelManager
from nataili.util.logger import logger

mm = CompVisModelManager()

for model in mm.models:
    logger.info(f"Downloading {model}...")
    mm.download_model(model)
