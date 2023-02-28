import uuid

from PIL import Image

from nataili.model_manager.compvis import CompVisModelManager
from nataili.stable_diffusion.compvis import CompVis
from nataili.util.monitor import VRAMMonitor, RAMMonitor
from nataili.util.logger import logger, set_logger_verbosity

set_logger_verbosity(5)

vram = VRAMMonitor(device_id=0)
ram = RAMMonitor()

logger.info(f"VRAM total {vram.total} {vram.unit}")
logger.info(f"RAM total {ram.total} {ram.unit}")

vram.start()
ram.start()

logger.info(f"VRAM start {vram.start_usage} {vram.unit}")
logger.info(f"RAM start {ram.start_usage} {ram.unit}")
mm = CompVisModelManager()

model = "stable_diffusion"

mm.load(model, gpu_id=0)
vram.stop()
ram.stop()
logger.info(f"VRAM peak {vram.get_peak()} {vram.unit}")
logger.info(f"RAM peak {ram.get_peak()} {ram.unit}")

logger.info(f"VRAM end {vram.end_usage} {vram.unit}")
logger.info(f"RAM end {ram.end_usage} {ram.unit}")


output_dir = f"./test_output/{str(uuid.uuid4())}"

prompt = "(stylized art:0.75), (Dungeons and Dragons:0.9), male (half-orc:1.1) barbarian, (wearing shorts and leather boots:1.05), (bald:1.1), (gray skin color:1.1), (holding a giant axe:1.2), (in the forest:1.1), (Kratos vibes:0.85), (Brock Lesnar vibes:0.95), (Drax vibes:0.9), wild beastly demeanor, (badass brute:1.05), art by Vox Machina and Code Lyoko and Maximum Ride, cel shaded, cinematic, 16k ### (low quality:0.85), bad proportions, (ornate:0.85), collage, multiple photos, text, watermark, group photo, crowded, mosh pit, yellow skin, (achromatic:0.75) ### weird mouth,weird teeth,incorrect anatomy, bad anatomy, ugly, odd, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, bad drawing, sketch, disfigured, cropped, out of frame,fully clothed,sad,agony,pain,cut off,trimmed,headpiece, head necklace,out of frame, malformed,hideous face, too many limbs, missing fingers, too many fingers, text, bad drawing, sketch, incorrect anatomy, hideous, unsightly, extra limbs, poorly drawn, poorly drawn face, disfigured, cropped"

vram.start()
ram.start()
logger.info(f"VRAM start {vram.start_usage} {vram.unit}")
logger.info(f"RAM start {ram.start_usage} {ram.unit}")
compvis = CompVis(
    model=mm.loaded_models[model],
    model_name=model,
    output_dir=output_dir,
    disable_voodoo=True,
)
compvis.generate(
    prompt,
)
vram.stop()
ram.stop()
logger.info(f"VRAM peak {vram.get_peak()} {vram.unit}")
logger.info(f"RAM peak {ram.get_peak()} {ram.unit}")

logger.info(f"VRAM end {vram.end_usage} {vram.unit}")
logger.info(f"RAM end {ram.end_usage} {ram.unit}")