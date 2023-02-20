import uuid
from PIL import Image
from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.controlnet import ControlNetModelManager
from nataili.stable_diffusion.compvis import CompVis

compvis = CompVisModelManager()
controlnet = ControlNetModelManager(download_reference=False)

init_image = Image.open(
    "./test_imgs/bird.png"
).convert("RGB")

sd_model_name = "Anything Diffusion"
compvis.load(sd_model_name, cpu_only=True)
anything_state_dict = compvis.loaded_models[sd_model_name]["model"].state_dict()

controlnet.load_controlnet("control_canny")

controlnet.load_control_ldm("control_canny", sd_model_name, anything_state_dict)

loaded_control_ldm = list(controlnet.loaded_models.keys())[0]
print(loaded_control_ldm)
output_dir = f"./test_output/{str(uuid.uuid4())}"
from nataili.stable_diffusion.compvis import CompVis
generator = CompVis(
    model=controlnet.loaded_models[loaded_control_ldm],
    model_baseline=compvis.models[sd_model_name]["baseline"],
    output_dir=output_dir,
    disable_voodoo=True,
    safety_checker=None,
    filter_nsfw=False,
)
generator.generate(
    prompt="1girl bird",
    init_img=init_image,
    control_type="canny",
    cfg_scale=9.0,
)
# display(generator.images[0]['image'])