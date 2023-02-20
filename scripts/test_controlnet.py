from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.controlnet import ControlNetModelManager

compvis = CompVisModelManager()
controlnet = ControlNetModelManager(download_reference=False)

sd_model_name = "Anything Diffusion"
compvis.load(sd_model_name, cpu_only=True)
anything_state_dict = compvis.loaded_models[sd_model_name]["model"].state_dict()

controlnet.load_controlnet("control_canny")

controlnet.load_control_ldm("control_canny", sd_model_name, anything_state_dict)

print(controlnet.loaded_models.keys())