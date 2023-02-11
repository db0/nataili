# Overview

```
NATAILI\
├───aitemplate
├───blip
├───cache
├───clip
│   └───predictor
├───codeformers
├───esrgan
├───gfpgan
├───model_manager
├───sdu
├───stable_diffusion
│   └───diffusers
├───train
│   ├───dataset
│   │   └───EveryDream
│   ├───dreambooth
│   └───lora
└───util
    ├───blip
    ├───codeformer
    └───gfpgan
```

### Model types

* `aitemplate`
* `blip`
* `clip`
    * `predictor`
* `codeformers`
* `esrgan`
* `gfpgan`
* `stable_diffusion`
    * `diffusers`
* `train`
    * `dataset`
        * `EveryDream`
    * `dreambooth`
    * `lora`

Generally speaking, each model type has a folder with the same name and related files in that folder.

Sometimes, a model type has a subfolder for a subtype of the model type. For example, the `clip` model type.

```
NATAILI\CLIP
├───coca.py # LAION-CoCa
├───image.py # Image embedding
├───interrogate.py # Interrogation / similarity
├───text.py # Text embedding
└───predictor
    ├───inference.py # Predictor inference
    ├───mlp.py # MLP for predictor
    ├───prepare.py # Preparing dataset for predictor training
    └───train.py # Predictor training
```

* LAION-CoCa is a model that uses CLIP, so it is in the `clip` folder.
* CLIP Image and text embedding functions are in the `image.py` and `text.py` files, respectively.
* The `interrogate.py` file contains the interrogation and similarity functions.

* The `predictor` folder contains several modules that are used for the CLIP+MLP predictor. These are grouped together in a folder because they are all related to the predictor.

Other examples of this are the `diffusers` folder in the `stable_diffusion` model type, and the `train` folder.

* Nataili mainly supports CompVis so `stable_diffusion` contains `compvis.py`
* Depth2img and Inpainting models are implemented via diffusers pipelines, so they are in the `diffusers` folder. `depth2img.py` and `inpainting.py` respectively.

```
NATAILI\TRAIN
├───dataset
│   └───EveryDream
├───dreambooth
│   └───dreambooth_lora.py
└───lora
    └───lora.py
```

* `dataset` contains additional dataloaders that can be used for training.
* `dreambooth` contains the Dreambooth training scripts, e.g. `dreambooth_lora.py` - this is Dreambooth with LoRA (and EveryDream). Others can be implemented here e.g. `dreambooth_diffusers.py` - this could be a version resembling the diffusers example script. Or a version with a different dataset e.g. `dreambooth_lora` without EveryDream.
* `lora` is everything related to LoRA.

* `train` would be expanded with other types of training e.g. `train\textual_inversion` for textual inversion training or `train\pivotal_tuning` for pivotal tuning training. LoRA versions of these would be under their respective folders and named accordingly e.g. `textual_inversion_lora.py` or `pivotal_tuning_lora.py`.

### Model Manager

```
NATAILI\MODEL_MANAGER
├───aitemplate.py
├───base.py
├───blip.py
├───clip.py
├───codeformer.py
├───compvis.py
├───diffusers.py
├───esrgan.py
├───gfpgan.py
├───new.py
├───safety_checker.py
├───sdu.py
└───super.py
```

* Each model type has a corresponding file in the `model_manager` folder. These files contain the model manager class for that model type.
* `base.py` contains the base model manager class. This is the parent class for all model manager classes. It contains the basic functions that all model managers can use.
* `new.py` is a template for creating a new model manager class. It contains the basic structure of a model manager class.
    * Copy this file and rename it to the name of the model type you want to create a model manager class for.
    * ctrl+f `new` and replace each instance with the name of the model type.
    * Now you can modify the `load` and `load_model` functions for the model type.
    * General structure of `load` and `load_model` should remain the same
        * `load` checks if the model is valid, if it is available, and downloads the model if it is not available.#
        * `load` calls `load_model` which loads the model and returns it.
        * `load_model` will override device/precision as needed i.e. CUDA unavailable, or precision is not supported.
        * The model is then stored in the Model Manager class instance under `self.loaded_model[model_name]`.
* `super.py` contains the super model manager class. This was created for backwards compatibility with v0.0.1/Stable Horde worker, as such it has a different structure to the other model manager classes, and does not have all the functions that the other model manager classes have.
* A model manager class can handle subtypes of a model type.
    * `clip` model manager class handles models with type `"clip"`, `"open_clip"` and `"coca"`, these are loaded in different ways so they are handled by different functions.
    * This type is defined in the model's database .json file. `"type": "clip"` for example.
