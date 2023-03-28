import uuid
import os

from PIL import Image

from nataili.model_manager.compvis import CompVisModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.logger import logger

init_image = Image.open(
    os.path.join(os.path.dirname(__file__), "inpaint_original.png")
).convert("RGB")

init_mask = Image.open(
    os.path.join(os.path.dirname(__file__), "inpaint_mask.png")
).convert("RGB")

mm = CompVisModelManager()

# model = "Epic Diffusion Inpainting"
model = "Deliberate Inpainting"

mm.load(model)


output_dir = f"./test_output/{str(uuid.uuid4())}"

prompt = "a cute puppy with sunglasses sitting on the bench, highly detailed, sharp focus, award winning photo, unreal engine 5, raytracing ### weird mouth,weird teeth,incorrect anatomy, bad anatomy, ugly, odd, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, bad drawing, sketch, disfigured, cropped, out of frame,fully clothed,sad,agony,pain,cut off,trimmed,headpiece, head necklace,out of frame, malformed,hideous face, too many limbs, missing fingers, too many fingers, text, bad drawing, sketch, incorrect anatomy, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, disfigured, cropped"

compvis = CompVis(
    model=mm.loaded_models[model],
    model_name=model,
    output_dir=output_dir,
    disable_voodoo=True,
)
compvis.generate(
    prompt,
    init_img=init_image,
    init_mask=init_mask,
    sampler_name="DDIM",
    inpainting=True,
)
