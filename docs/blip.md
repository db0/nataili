# Example Usage Script: BLIP
In this example, we will demonstrate how to use Nataili to generate captions for an image using BLIP.

### Import Required Libraries
We start by importing the necessary libraries:

```python
from PIL import Image
from nataili.blip import Caption
from nataili.model_manager.blip import BlipModelManager
from nataili.util.logger import logger
```
* `Image` from the Python Imaging Library (PIL) is used to open the image file.
* `Caption` from the `nataili.blip` module is the service class used to generate captions.
* `BlipModelManager` from the `nataili.model_manager.blip` module is the model manager used to download and manage the BLIP model.
* `logger` from the `nataili.util.logger` module is used to log the generated caption.

### Open the Image
Next, we open the image file:

```python
image = Image.open("01.png")
```

Replace `"01.png"` with the path to the image file you want to use.

### Create an Instance of the Model Manager
We create an instance of the `BlipModelManager`:

```python
mm = BlipModelManager()
```

### Load the BLIP Model
We use the load function of the model manager to download and load the BLIP model:

```python
mm.load("BLIP")
```
The `"BLIP"` argument is the key used to identify the BLIP model in the model database.

### Create a Caption Object
We create a `Caption` object, passing in the loaded BLIP model:

```python
blip = Caption(mm.loaded_models["BLIP"])
```

### Generate the Caption
Finally, we use the `__call__` function of the `Caption` object to generate the caption:

```python
logger.generation(f"caption: {blip(image)}")
```
The generated caption will be logged using the `logger` object. The format of the log message is `"caption: [generated caption]"`.

With these steps, we have successfully used Nataili to generate captions for an image using BLIP.