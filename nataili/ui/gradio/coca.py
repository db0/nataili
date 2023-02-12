import gradio as gr
from PIL import Image

from nataili.clip import CoCa
from nataili.model_manager.clip import ClipModelManager
from nataili.ui.gradio.base import GradioUI


class CoCaUI(GradioUI):
    def __init__(self):
        model_manager_class = ClipModelManager
        model_name = "coca_ViT-L-14"
        inputs = gr.Image(image_mode="RGB", type="pil", source="upload", label="Image")
        outputs = gr.Textbox(lines=1, label="Caption")
        title = "Nataili - Image Captioning with LAION-CoCa"
        description = "Image Captioning with LAION-CoCa"
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
        self.coca = None

    def init(self):
        if self.coca is None:
            self.coca = CoCa(
                self.model_manager.loaded_models[self.model_name]["model"],
                self.model_manager.loaded_models[self.model_name]["transform"],
                self.model_manager.loaded_models[self.model_name]["device"],
                self.model_manager.loaded_models[self.model_name]["half_precision"],
            )

    def predict(self, image: Image.Image):
        self.init()
        return self.coca(image)

    def __call__(self):
        self.launch(self.predict)


def main():
    coca_ui = CoCaUI()
    coca_ui()


if __name__ == "__main__":
    main()
