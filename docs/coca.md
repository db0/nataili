# Example Usage Script: Interrogating Images with LAION-CoCa

In this example, we will demonstrate how to use LAION-CoCa to generate captions for an image.

## Import Required Libraries

We start by importing the necessary libraries:

```python
from PIL import Image
from nataili.clip import CoCa
from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger
```

`Image` from the Python Imaging Library (PIL) is used to open the image files.
`CoCa` from the `nataili.clip` module is the service class used to generate captions for the images.
`ClipModelManager` from the `nataili.model_manager.clip` module is the model manager used to download and manage the CLIP model.
`logger` from the `nataili.util.logger` module is used to log the results.

### Open the Image
Next, we open the image file:

```python
image = Image.open("01.png")
```

Replace `"01.png"` with the path to the image file you want to use.

### Create an Instance of the Model Manager

We create an instance of the `ClipModelManager`:

```python
mm = ClipModelManager()
```

### Load the CoCa Model
We use the `load` function of the model manager to download and load the CoCa model:

```python
mm.load("coca_ViT-L-14")
```
The `"coca_ViT-L-14"` argument is the key used to identify the CoCa model in the model database.

### Create a CoCa Object
We create a `CoCa` object, passing in the loaded CoCa model and related information:

```python
coca = CoCa(
    mm.loaded_models["coca_ViT-L-14"]["model"],
    mm.loaded_models["coca_ViT-L-14"]["transform"],
    mm.loaded_models["coca_ViT-L-14"]["device"],
    mm.loaded_models["coca_ViT-L-14"]["half_precision"],
)
```

### Generate the Caption
We use the `__call__` function of the `CoCa` object to generate the caption for the image:

```python
logger.generation(coca(image))
```
The results will be logged using the `logger` object. The format of the log message is `"caption: [generated caption]"`.

With these steps, we have successfully used LAION-CoCa to generate captions for an image.