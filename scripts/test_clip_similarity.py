import os

import PIL

from nataili import Interrogator, ModelManager, logger

images = []

directory = "teeth"

for file in os.listdir(directory):
    pil_image = PIL.Image.open(f"{directory}/{file}").convert("RGB")
    images.append({"pil_image": pil_image, "filename": file})

mm = ModelManager()

mm.clip.load("ViT-L/14")

interrogator = Interrogator(
    mm.clip.loaded_models["ViT-L/14"],
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


for image in images:
    results = interrogator(image['pil_image'], word_list, top_count=5, similarity=True)
    html_string += f"""
    <h1>{image['filename']}</h1>
    <img src="{directory}/{image['filename']}" width="300" />
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
