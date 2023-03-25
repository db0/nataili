# Nataili: Multimodal AI Python Library

[![PyPI version](https://badge.fury.io/py/nataili.svg)](https://badge.fury.io/py/nataili)
[![Downloads](https://pepy.tech/badge/nataili)](https://pepy.tech/project/nataili)

![GitHub license](https://img.shields.io/github/license/Sygil-Dev/nataili)

Nataili is a Python library that provides tools for building multimodal AI applications. With its modular design, Nataili makes it easy to use only the tools you need to build custom AI solutions.

Some of the technologies included in Nataili are:

* AITemplate: Fast diffusion
* BLIP: Image captioning
* CLIP: Interrogation (ranking, similarity), Embedding (text, image) with cache support
* CLIP+MLP Predictor
* CodeFormer: Post-processing and super-resolution for faces
* ESRGAN: Image super-resolution
* GFPGAN: Post-processing for faces
* LAION-CoCa: Image captioning
* Stable Diffusion: Image generation
* SD Upscaler: latent diffusion upscaler for Stable Diffusion's autoencoder
* Train: Training tools for Stable Diffusion and CLIP+MLP Predictor

## Projects using Nataili

* [AI Horde worker](https://github.com/db0/AI-Horde-Worker)

## Installation

`pip install nataili`

From source:

```
git clone https://github.com/db0/nataili
cd nataili
pip install -e .
```

## Documentation

* [Index](docs/README.md)
* [Overview](docs/overview.md)
* [Model Manager](docs/model_manager.md)
* [Blip](docs/blip.md)
* [Clip](docs/clip.md)
* [LAION-CoCa](docs/coca.md)
