from typing import Any, Callable, Dict, List, Union

import gradio as gr

from nataili.model_manager.blip import BlipModelManager
from nataili.model_manager.clip import ClipModelManager
from nataili.model_manager.codeformer import CodeFormerModelManager
from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.diffusers import DiffusersModelManager
from nataili.model_manager.esrgan import EsrganModelManager
from nataili.model_manager.gfpgan import GfpganModelManager
from nataili.model_manager.safety_checker import SafetyCheckerModelManager


class GradioUI:
    def __init__(
        self,
        inputs,
        outputs,
        model_manager_class: Union[
            BlipModelManager,
            ClipModelManager,
            CodeFormerModelManager,
            CompVisModelManager,
            DiffusersModelManager,
            EsrganModelManager,
            GfpganModelManager,
            SafetyCheckerModelManager,
        ] = None,
        model_name: str = "",
        title: str = "",
        description: str = "",
        thumbnail: str = "",
    ):
        self.model_manager_class = model_manager_class
        self.model_name = model_name
        self.inputs = inputs
        self.outputs = outputs
        self.title = title
        self.description = description
        self.thumbnail = thumbnail
        self.model_manager = None

    def launch(self, predict_function: Callable):
        if self.model_manager_class is not None and self.model_name != "":
            self.model_manager = self.model_manager_class()
            self.model_manager.load(self.model_name)

        interface = gr.Interface(
            predict_function,
            self.inputs,
            self.outputs,
            title=self.title,
            description=self.description,
            thumbnail=self.thumbnail,
            analytics_enabled=False,
            allow_flagging="never",
        )

        interface.launch(show_error=True, quiet=False, debug=True)
