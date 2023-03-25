import os

from setuptools import find_packages, setup

os.environ["PIP_EXTRA_INDEX_URL"] = "https://download.pytorch.org/whl/cu117"

requirements = []
with open("requirements.txt") as reqstxt:
    requirements = reqstxt.readlines()

setup(
    name="nataili",
    version="v0.2.9019",
    description="",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nataili_ui_coca = nataili.ui.gradio.coca:main",
            "nataili_ui_blip = nataili.ui.gradio.blip:main",
            "nataili_ui_codeformer = nataili.ui.gradio.codeformer:main",
        ]
    },
)
