import requests
from PIL import Image

from nataili.model_manager.diffusers import DiffusersModelManager
from nataili.stable_diffusion.diffusers.depth2img import Depth2Img
from nataili.util.logger import logger

mm = DiffusersModelManager(download_reference=False)

model = "Stable Diffusion 2 Depth"

success = mm.load(model)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)

generator = Depth2Img(
    pipe=mm.loaded_models[model]["model"],
    device=mm.loaded_models[model]["device"],
    output_dir="bridge_generations",
    filter_nsfw=False,
    disable_voodoo=True,
)

prompt = "two tigers ### bad, deformed, ugly, bad anatomy"
for iter in range(5):
    output = generator.generate(prompt=prompt, init_img=init_image, height=480, width=640)
