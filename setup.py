import os

from setuptools import find_packages, setup

os.environ["PIP_EXTRA_INDEX_URL"] = "https://download.pytorch.org/whl/cu117"


setup(
    name="nataili",
    version="0.2.46",
    description="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "k-diffusion",
        "omegaconf",
        "diffusers",
        "fairscale",
        "tqdm",
        "python-slugify",
        "einops",
        "facexlib",
        "kornia",
        "opencv-python-headless",
        "basicsr",
        "gfpgan",
        "realesrgan",
        "loguru",
        "pydantic",
        "bitsandbytes",
        "transformers",
        "open-clip-torch",
        "pytorch-lightning",
        "accelerate",
        "ray",
    ],
    entry_points={
        "console_scripts": [
            "nataili_ui_coca = nataili.ui.gradio.coca:main",
            "nataili_ui_blip = nataili.ui.gradio.blip:main",
            "nataili_ui_codeformer = nataili.ui.gradio.codeformer:main",
        ]
    },
)
