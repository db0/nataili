import uuid

from PIL import Image

from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.safety_checker import SafetyCheckerModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.logger import logger

init_image = Image.open("./01.png").convert("RGB")

mm = CompVisModelManager()
smm = SafetyCheckerModelManager()

model = "stable_diffusion"
safety_model = "safety_checker"

def test_compvis(
    model,
    prompt,
    sampler,
    steps=30,
    output_dir=None,
    seed=None,
    init_img=None,
    denoising_strength=0.75,
    sigma_override=None,
    filter_nsfw=False,
    safety_checker=None,
):
    log_message = f"sampler: {sampler} steps: {steps} model: {model}"
    if seed:
        log_message += f" seed: {seed}"
    if sigma_override:
        log_message += f" sigma_override: {sigma_override}"
    logger.info(log_message)
    if filter_nsfw:
        logger.info("Filtering NSFW")
    if init_img:
        logger.info("Using init image")
        logger.info(f"Denoising strength: {denoising_strength}")
    compvis = CompVis(
        model=mm.loaded_models[model],
        model_name=model,
        output_dir=output_dir,
        disable_voodoo=True,
        filter_nsfw=filter_nsfw,
        safety_checker=safety_checker,
    )
    compvis.generate(
        prompt,
        sampler_name=sampler,
        ddim_steps=steps,
        seed=seed,
        init_img=init_img,
        sigma_override=sigma_override,
        denoising_strength=denoising_strength,
    )


samplers = [
    "k_lms",
]

output_dir = f"./test_output/{str(uuid.uuid4())}"


@logger.catch
def test():
    mm.load(model)
    smm.load(safety_model)
    logger.info(f"Output dir: {output_dir}")
    logger.debug(f"Running inference on {model}")
    logger.info(f"Testing safety checker")
    prompt = (
        "boobs"
    )
    logger.info(f"Prompt: {prompt}")
    for sampler in samplers:
        test_compvis(model, prompt, sampler, output_dir=output_dir, safety_checker=smm.loaded_models[safety_model]["model"], filter_nsfw=True)

if __name__ == "__main__":
    test()
