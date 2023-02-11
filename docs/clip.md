# Example Usage Script: Interrogating Images with CLIP

In this example, we will demonstrate how to use CLIP to interrogate a set of images with a list of words, and rank the words based on how well they match the images relative to each other.

## Import Required Libraries

We start by importing the necessary libraries:

```python
import os
from PIL import Image
from nataili.model_manager.clip import ClipModelManager
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger
```

`os` is used to list the files in a directory.
`Image` from the Python Imaging Library (PIL) is used to open the image files.
`ClipModelManager` from the `nataili.model_manager.clip` module is the model manager used to download and manage the CLIP model.
`Interrogator` from the `nataili.clip` module is the service class used to interrogate the images.
`logger` from the `nataili.util.logger` module is used to log the results.

## Define the Images Directory

Next, we define the directory containing the images:

```python
images = []

directory = "test_images"
```

Replace `"test_images"` with the path to the directory containing the images you want to interrogate.

## Create an Instance of the Model Manager

We create an instance of the `ClipModelManager`:

```python
mm = ClipModelManager()
```

## Load the CLIP Model

We use the `load` function of the model manager to download and load the CLIP model:

```python
mm.load("ViT-L/14")
```

The `"ViT-L/14"` argument is the key used to identify the CLIP model in the model database.

## Create an Interrogator Object

We create an `Interrogator` object, passing in the loaded CLIP model:

```python
interrogator = Interrogator(
    mm.loaded_models["ViT-L/14"],
)
```

## Interrogate the Images

We use the `__call__` function of the `Interrogator` object to interrogate the images:

```python
for file in os.listdir(directory):
    image  = Image.open(f"{directory}/{file}").convert("RGB")
    results = interrogator(image=image, text_array=None, rank=True, top_count=5)
    """
    or
    results = interrogator(filename=file, directory=directory, text_array=None, rank=True, top_count=5)
    """
    logger.generation(results)
```

The `__call__` function accepts an image, represented either as an `Image` object or as a file path, and returns the most similar images. The `rank` argument, when set to True, returns the results ranked by similarity. The `top_count` argument specifies the number of results to return.

The results will be logged using the `logger` object. The format of the log message is `"results: [interrogation results]"`.

With these steps, we have successfully used Nataili to interrogate a set of images with a list of words, and rank the words based on how well they match the images relative to each other.
