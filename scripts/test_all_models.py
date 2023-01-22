from uuid import uuid4

from nataili.model_manager.compvis import CompVisModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.logger import logger

mm = CompVisModelManager()
base_dir = f"./test_output/{str(uuid4())}"
logger.info(f"Output dir: {base_dir}")
prompt = (
    "Headshot of cybernetic female character, cybernetic implants, solid background color,"
    "digital art, illustration, smooth color, cinematic moody lighting, cyberpunk, body modification,"
    "wenjun lin, studio ghibli, pixiv, artgerm, greg rutkowski, ilya kuvshinov"
)

for model in mm.models:
    output_dir = f"{base_dir}/{model}"
    logger.init(f"{model}", status="Loading")
    mm.load(model)
    logger.info(f"Running inference on {model}")
    compvis = CompVis(model=mm.loaded_models[model], output_dir=output_dir, model_name=model, disable_voodoo=True)
    compvis.generate(prompt)
    del mm.loaded_models[model]
