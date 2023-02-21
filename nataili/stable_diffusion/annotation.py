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
import cv2
import einops
import numpy as np
import torch
from PIL import Image

from annotator.canny import CannyDetector
from annotator.hed import HEDdetector, nms
from annotator.midas import MidasDetector
from annotator.mlsd import MLSDdetector
from annotator.openpose import OpenposeDetector
from annotator.uniformer import UniformerDetector
from annotator.util import HWC3, resize_image


class Annotation:
    def __init__(self):
        self.model = None


class Canny(Annotation):
    def __init__(self):
        super().__init__()
        self.model = CannyDetector()

    def __call__(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
        resolution: int = 512,
        num_samples: int = 1,
    ):
        image = np.asarray(image)
        image = resize_image(HWC3(image), resolution)
        H, W, C = image.shape
        detected_map = self.model(image, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": 255 - detected_map, "shape": (H, W, C)}


class HED(Annotation):
    def __init__(self):
        super().__init__()
        self.model = HEDdetector()

    def __call__(self, image: Image.Image, resolution: int = 512, detect_resolution: int = 512, num_samples: int = 1):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map = self.model(resize_image(image, detect_resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class FakeScribbles(Annotation):
    def __init__(self):
        super().__init__()
        self.model = HEDdetector()

    def __call__(self, image: Image.Image, resolution: int = 512, detect_resolution: int = 512, num_samples: int = 1):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map = self.model(resize_image(image, detect_resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Hough(Annotation):
    def __init__(self):
        super().__init__()
        self.model = MLSDdetector()

    def __call__(
        self,
        image: Image.Image,
        value_threshold: float = 0.1,
        distance_threshold: float = 0.1,
        resolution: int = 512,
        detect_resolution: int = 512,
        num_samples: int = 1,
    ):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map = self.model(resize_image(image, detect_resolution), value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Depth(Annotation):
    def __init__(self):
        super().__init__()
        self.model = MidasDetector()

    def __call__(self, image: Image.Image, resolution: int = 512, depth_resolution: int = 384, num_samples: int = 1):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map, _ = self.model(resize_image(image, depth_resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Normal(Annotation):
    def __init__(self):
        super().__init__()
        self.model = MidasDetector()

    def __call__(
        self,
        image: Image.Image,
        background_threshold: float = 0.4,
        resolution: int = 512,
        depth_resolution: int = 384,
        num_samples: int = 1,
    ):
        image = np.asarray(image)
        image = HWC3(image)
        _, detected_map = self.model(resize_image(image, depth_resolution), bg_th=background_threshold)
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Openpose(Annotation):
    def __init__(self):
        super().__init__()
        self.model = OpenposeDetector()

    def __call__(
        self,
        image: Image.Image,
        has_hand: bool = False,
        resolution: int = 512,
        detect_resolution: int = 512,
        num_samples: int = 1,
    ):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map, _ = self.model(resize_image(image, detect_resolution), has_hand)
        detected_map = HWC3(detected_map)
        image = resize_image(image, resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Seg(Annotation):
    def __init__(self):
        super().__init__()
        self.model = UniformerDetector()

    def __call__(self, image: Image.Image, resolution: int = 512, detect_resolution: int = 512, num_samples: int = 1):
        image = np.asarray(image)
        image = HWC3(image)
        detected_map = self.model(resize_image(image, detect_resolution))
        image = resize_image(image, resolution)
        H, W, C = image.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}


class Scribble(Annotation):
    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, image: Image.Image, resolution: int = 512, num_samples: int = 1):
        image = np.asarray(image)
        image = resize_image(HWC3(image), resolution)
        H, W, C = image.shape
        detected_map = np.zeros_like(image, dtype=np.uint8)
        detected_map[np.min(image, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        return {"control": control, "detected_map": detected_map, "shape": (H, W, C)}
