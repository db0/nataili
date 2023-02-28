import uuid

from PIL import Image

from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.controlnet import ControlNetModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.logger import logger, set_logger_verbosity


set_logger_verbosity(5)
init_image = Image.open(
    "./test_images/controlnet/canny.png"
).convert("RGB")
cnmm = ControlNetModelManager(download_reference=False)
mm = CompVisModelManager()

model = "Anything Diffusion"

mm.load(model)


output_dir = f"./test_output/{str(uuid.uuid4())}"

prompt = "anime girl ### weird mouth,weird teeth,incorrect anatomy, bad anatomy, ugly, odd, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, bad drawing, sketch, disfigured, cropped, out of frame,fully clothed,sad,agony,pain,cut off,trimmed,headpiece, head necklace,out of frame, malformed,hideous face, too many limbs, missing fingers, too many fingers, text, bad drawing, sketch, incorrect anatomy, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, disfigured, cropped"


control_types = ["canny"]

for control_type in control_types:
    logger.info(f"Testing control_type='{control_type}'")
    compvis = CompVis(
        model=mm.loaded_models[model],
        model_baseline=mm.models[model]["baseline"],
        model_name=model,
        output_dir=output_dir,
        disable_voodoo=True,
        control_net_manager=cnmm
    )
    compvis.generate(
        prompt,
        init_img=init_image,
        control_type=control_type,
        init_as_control=True,
        ddim_steps=20,
    )
