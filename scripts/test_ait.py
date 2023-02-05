from nataili.aitemplate import AITemplate
from nataili.model_manager.aitemplate import AITemplateModelManager
from nataili.util.logger import logger

mm = AITemplateModelManager()

mm.load()


def run():
    while True:
        logger.info("init")
        ait = AITemplate(mm.loaded_models["ait"])
        logger.info("start")
        ait.generate("corgi", ddim_steps=30)
        logger.info("end")


if __name__ == "__main__":
    run()
