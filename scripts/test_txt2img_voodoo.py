import uuid

from PIL import Image

from nataili.model_manager.compvis import CompVisModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.logger import logger

mm = CompVisModelManager()

model = "stable_diffusion"

mm.load(model, voodoo=True)


output_dir = f"./test_output/{str(uuid.uuid4())}"

prompt = "(stylized art:0.75), (Dungeons and Dragons:0.9), male (half-orc:1.1) barbarian, (wearing shorts and leather boots:1.05), (bald:1.1), (gray skin color:1.1), (holding a giant axe:1.2), (in the forest:1.1), (Kratos vibes:0.85), (Brock Lesnar vibes:0.95), (Drax vibes:0.9), wild beastly demeanor, (badass brute:1.05), art by Vox Machina and Code Lyoko and Maximum Ride, cel shaded, cinematic, 16k ### (low quality:0.85), bad proportions, (ornate:0.85), collage, multiple photos, text, watermark, group photo, crowded, mosh pit, yellow skin, (achromatic:0.75) ### weird mouth,weird teeth,incorrect anatomy, bad anatomy, ugly, odd, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, bad drawing, sketch, disfigured, cropped, out of frame,fully clothed,sad,agony,pain,cut off,trimmed,headpiece, head necklace,out of frame, malformed,hideous face, too many limbs, missing fingers, too many fingers, text, bad drawing, sketch, incorrect anatomy, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, disfigured, cropped"

compvis = CompVis(
    model=mm.loaded_models[model],
    model_name=model,
    output_dir=output_dir,
    disable_voodoo=False,
)
compvis.generate(
    prompt,
)
