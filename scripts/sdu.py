from PIL import Image

from nataili.model_manager.sdu import SDUModelManager
from nataili.sdu import StableDiffusionUpscaler

mm = SDUModelManager()

mm.load("Stable Diffusion Upscaler")

sdu = StableDiffusionUpscaler(
    model=mm.loaded_models["Stable Diffusion Upscaler"]["model"],
    vae_840k=mm.loaded_models["Stable Diffusion Upscaler"]["vae_model_840k"],
    vae_560k=mm.loaded_models["Stable Diffusion Upscaler"]["vae_model_560k"],
    tokenizer=mm.loaded_models["Stable Diffusion Upscaler"]["tokenizer"],
    text_encoder=mm.loaded_models["Stable Diffusion Upscaler"]["text_encoder"],
    device=mm.loaded_models["Stable Diffusion Upscaler"]["device"]
)

input_image = Image.open("sd_2x_upscaler_demo.png").convert("RGB")

sdu(
    prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas",
    input_image = input_image,
)
    