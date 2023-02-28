import os

from PIL import Image

from nataili.model_manager.clip import ClipModelManager
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger

images = []

directory = "teeth"

mm = ClipModelManager()

mm.load("ViT-L/14")

interrogator = Interrogator(
    mm.loaded_models["ViT-L/14"],
)

base_words = [
    'teeth',
    # 'hand',
    # 'mouth',
    # 'fingers',
]

description_words = [
    'ugly',
    # 'bad',
    # 'misshaped',
    # 'misfigured',
    # 'malformed',
    # 'deformed',
    # 'disfigured',
    # 'disabled',
    # 'good',
    # 'beautiful',
    # 'perfect',
    # 'well-formed',
    # 'nice',
    'pretty',
]

word_list = []

for base_word in base_words:
    for description_word in description_words:
        word_list.append(f"{description_word} {base_word}")
    word_list.append(base_word) 

html_string = ""



for file in os.listdir(directory):
    results = interrogator(filename=file, directory=directory, text_array=word_list, similarity=True)
    results = results['default']
    html_string += f"""
    <h1>{file}</h1>
    <img src="{directory}/{file}" width="300" />
    <table>
    <tr>
    <th>Word</th>
    <th>Score</th>
    </tr>
    """
    for key in results.keys():
        html_string += f"""
        <tr>
        <td>{key}</td>
        <td>{results[key]}</td>
        </tr>
        """
    html_string += "</table>"
    html_string += "<hr />"

with open("test.html", "w") as f:
    f.write(html_string)
