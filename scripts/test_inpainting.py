from PIL import Image

from nataili.model_manager.diffusers import DiffusersModelManager
from nataili.stable_diffusion.diffusers.inpainting import inpainting
from nataili.util.logger import logger

mm = DiffusersModelManager(download_reference=False)

model = "stable_diffusion_inpainting"

success = mm.load(model)

original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")

generator = inpainting(
    pipe=mm.loaded_models[model]["model"],
    device=mm.loaded_models[model]["device"],
    output_dir="bridge_generations",
    filter_nsfw=False,
    disable_voodoo=True,
)
generator.generate("a mecha robot sitting on a bench", original, mask)
image = generator.images[0]["image"]
image.save("robot_sitting_on_a_bench.png", format="PNG")
