"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import contextlib
import copy
import glob
import os
import pickle
import shutil
import warnings
from typing import Dict, List, Tuple, TypeVar

import ray
import torch

from nataili import InvalidModelCacheException, enable_local_ray_temp, enable_ray_alternative
from nataili.aitemplate import Model
from nataili.util.logger import logger

warnings.filterwarnings("ignore")


T = TypeVar("T")


def get_ray_temp_dir():
    return os.path.abspath(os.environ.get("RAY_TEMP_DIR", "./ray"))


def get_model_cache_dir():
    return os.path.join(get_ray_temp_dir(), "model-cache")


def initialise_voodoo():
    if enable_local_ray_temp.active:
        ray_temp_dir = get_ray_temp_dir()
        session_dirs = glob.glob(os.path.join(ray_temp_dir, "session_*"))
        for adir in session_dirs:
            shutil.rmtree(adir, ignore_errors=True)
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(_temp_dir=ray_temp_dir)
        logger.init(f"Ray temp dir '{ray_temp_dir}'", status="Prepared")
    else:
        logger.init_warn("Ray temp dir'", status="OS Default")


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).cpu().detach().numpy() for name, param in module.named_parameters(recurse=False)
        }
        buffers = {name: torch.clone(buf).cpu().detach().numpy() for name, buf in module.named_buffers(recurse=False)}
        tensors.append({"params": params, "buffers": buffers})

    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device="cuda"):
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device), requires_grad=False),
            )
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))


def get_model_cache_filename(model_filename):
    return os.path.join(get_model_cache_dir(), os.path.basename(model_filename)) + ".cache"


def have_model_cache(model_filename):
    cache_file = get_model_cache_filename(model_filename)
    if os.path.exists(cache_file):
        # We have a cache file but only consider it valid if it's up to date
        model_timestamp = os.path.getmtime(model_filename)
        cache_timestamp = os.path.getmtime(cache_file)
        if model_timestamp <= cache_timestamp:
            return True
    return False


@contextlib.contextmanager
def load_from_plasma(ref, device="cuda"):
    if not enable_ray_alternative.active:
        # Load object from ray object store
        skeleton, weights = ray.get(ref)
    else:
        try:
            # Load object from our persistent store
            with open(ref, "rb") as cache:
                skeleton, weights = pickle.load(cache)
        except (pickle.PickleError, EOFError):
            # Most likely corrupt cache file, remove the file
            try:
                os.remove(ref)
            except OSError:
                pass  # we tried
            raise InvalidModelCacheException(f"Model .cache file {ref} was corrupt. It has been removed.")
    replace_tensors(skeleton, weights, device=device)
    skeleton.eval().to(device, memory_format=torch.channels_last)
    yield skeleton
    torch.cuda.empty_cache()


def push_model_to_plasma(model: torch.nn.Module, filename="") -> ray.ObjectRef:
    if not enable_ray_alternative.active:
        # Store object in ray object store
        ref = ray.put(extract_tensors(model))
    else:
        # Store object directly on disk
        cachefile = get_model_cache_filename(filename)
        if have_model_cache(cachefile):
            return cachefile
        # Create cache directory if it doesn't already exist
        if not os.path.isdir(get_model_cache_dir()):
            os.makedirs(get_model_cache_dir(), exist_ok=True)
        # Serialise our object
        with open(cachefile, "wb") as cache:
            pickle.dump(extract_tensors(model), cache, protocol=pickle.HIGHEST_PROTOCOL)
        ref = cachefile

    return ref


@contextlib.contextmanager
def load_diffusers_pipeline_from_plasma(ref, device="cuda"):
    pipe, modules = ray.get(ref)
    for name, weights in modules.items():
        replace_tensors(getattr(pipe, name), weights, device=device)
    pipe.to(device)
    yield pipe
    torch.cuda.empty_cache()


def push_diffusers_pipeline_to_plasma(pipe) -> ray.ObjectRef:
    modules = {}
    components = pipe.components
    for name, component in components.items():
        if isinstance(component, torch.nn.Module):
            skeleton, weights = extract_tensors(component)
            setattr(pipe, name, skeleton)
            modules[name] = weights
    ref = ray.put((pipe, modules))
    return ref


def init_ait_module(
    model_name,
    workdir,
):
    mod = Model(os.path.join(workdir, model_name))
    return mod


def push_ait_module(module: Model) -> ray.ObjectRef:
    ref = ray.put(module)
    return ref


def load_ait_module(ref):
    ait_module = ray.get(ref)
    return ait_module
