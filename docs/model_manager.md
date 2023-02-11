# Model Manager

### Key concepts

* Model managers are used to download and manage models for a specific service, the metadata for which are stored in a .json file and included in the package.
* The models are stored in the `NATAILI_CACHE_HOME/nataili/` directory or `XDG_CACHE_HOME/nataili/` directory (which defaults to `~/.cache/nataili`) and organized in a directory named after the service.
* `BaseModelManager` class provides basic functionality for managing models, including downloading models, listing models, validating models, and CUDA support detection.
* Service-specific classes inherit from `BaseModelManager` and offer a `load()` function for loading the model, which calls the internal `load_<service>()` function.
* Loaded models are stored in the `self.loaded_models` dictionary under the corresponding model key e.g. `self.loaded_models[model_key]`

### Model file structure

The structure of a model file in the Nataili Model Manager is defined in a JSON format, as shown in the example below:

```
{
  "": {
    "name": "",
    "type": "",
    "config": {
      "files": [
        {
          "path": ""
        }
      ],
      "download": [
        {
          "file_name": "",
          "file_path": "",
          "file_url": ""
        }
      ]
    },
    "available": false
  }
}
```

The fields in the model file structure are:

* `key`: A unique identifier for the model, used by the model manager to distinguish between different models.
* `name`: A descriptive name for the model, used to provide a human-readable description of the model.
* `type`: The type of model, which is used by the model manager to determine how the model should be processed.
* `config`: Configuration information for the model, including required files and download information.
* `files`: A list of files that are required for the model to work. The path `field` specifies the location of each file.
* `download`: Information about how to download the required files. The `file_name`, `file_path`, and `file_url` fields specify the name, location, and URL of each file, respectively.
* `available`: A boolean field indicating whether the model is available or not. The model manager sets this field to `true` if the model has been successfully downloaded.

### Basic usage

To use the Nataili Model Manager, follow these steps:

1. Import the service-specific model manager:
```python
from nataili import <service>ModelManager
```
2. Create an instance of the model manager:
```python
model_manager = <service>ModelManager()
```
3. Load the desired model using the load function and the model key:
```python
model_manager.load(<model_key>)
```

Where `<service>` is the name of the service, and `<model_key>` is the key used to identify the desired model in the model database.

### Service Classes
Service classes provide an interface for interacting with the models. To use a service class, follow these steps:

1. Load the desired model using the model manager.
2. Pass the loaded model as an argument to the service class.
3. Use the `__call__` function to interact with the model, accepting inputs and returning outputs.
