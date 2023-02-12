import gradio as gr
from PIL import Image

from nataili.codeformers import codeformers
from nataili.model_manager.codeformer import CodeFormerModelManager
from nataili.ui.gradio.base import GradioUI
from nataili.util.logger import logger


class CodeformerUI(GradioUI):
    def __init__(self):
        model_manager_class = CodeFormerModelManager
        model_name = "CodeFormers"
        inputs = gr.Image(label="Input", type="pil", image_mode="RGB", source="upload")
        outputs = "image"
        title = "Nataili - Face fixing and super resolution with CodeFormers"
        description = "Face fixing and super resolution with CodeFormers"
        thumbnail = ""
        super().__init__(
            model_manager_class=model_manager_class,
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            title=title,
            description=description,
            thumbnail=thumbnail,
        )
        self.codeformer = None

    def init(self):
        if self.codeformer is None:
            self.codeformer = codeformers(self.model_manager.loaded_models[self.model_name])

    def predict(self, image: Image.Image):
        self.init()
        return self.codeformer(input_image=image)

    def __call__(self):
        self.launch(self.predict)


def main():
    codeformer_ui = CodeformerUI()
    codeformer_ui()


if __name__ == "__main__":
    main()
