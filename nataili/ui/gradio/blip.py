import gradio as gr
from PIL import Image

from nataili.blip import Caption
from nataili.model_manager.blip import BlipModelManager
from nataili.ui.gradio.base import GradioUI


class BlipUI(GradioUI):
    def __init__(self):
        model_manager_class = BlipModelManager
        model_name = "BLIP_Large"
        inputs = gr.Image(image_mode="RGB", type="pil", source="upload", label="Image")
        outputs = gr.Textbox(lines=1, label="Caption")
        title = "Nataili - Image Captioning with BLIP"
        description = "Image Captioning with BLIP"
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
        self.caption = None

    def init(self):
        if self.caption is None:
            self.caption = Caption(self.model_manager.loaded_models[self.model_name])

    def predict(self, image: Image.Image):
        self.init()
        return self.caption(image)

    def __call__(self):
        self.launch(self.predict)


def main():
    blip_ui = BlipUI()
    blip_ui()


if __name__ == "__main__":
    main()
