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
import math
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

from nataili.util.postprocessor import PostProcessor

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


class esrgan(PostProcessor):
    def set_filename_append(self):
        """
        Set filename append
        """
        self.filename_append = "esrgan"

    def process(self, img, img_array, **kwargs):
        """
        Override process method from PostProcessor
        :param img: PIL Image
        :param img_array: numpy array
        :param kwargs: strength
        :return: PIL Image
        """
        with contextlib.redirect_stdout(None):
            try:
                output, _ = self.model["model"].enhance(img_array)
                output_array = np.array(output)
                output_image = Image.fromarray(output_array)
            except:
                output_image = self.esrgan_upscale(model=self.model["model"], input_img=img)
        return output_image

    def esrgan_upscale(self, model, input_img):
        def esrgan_enhance(model, img):
            img = np.array(img)
            img = img[:, :, ::-1]
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0).to(self.model["device"])
            with torch.no_grad():
                output = model(img)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = 255.0 * np.moveaxis(output, 0, 2)
            output = output.astype(np.uint8)
            output = output[:, :, ::-1]
            return Image.fromarray(output, "RGB")

        def split_grid(image, tile_w=512, tile_h=512, overlap=64):
            w = image.width
            h = image.height

            non_overlap_width = tile_w - overlap
            non_overlap_height = tile_h - overlap

            cols = math.ceil((w - overlap) / non_overlap_width)
            rows = math.ceil((h - overlap) / non_overlap_height)

            dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
            dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

            grid = Grid([], tile_w, tile_h, w, h, overlap)
            for row in range(rows):
                row_images = []

                y = int(row * dy)

                if y + tile_h >= h:
                    y = h - tile_h

                for col in range(cols):
                    x = int(col * dx)

                    if x + tile_w >= w:
                        x = w - tile_w

                    tile = image.crop((x, y, x + tile_w, y + tile_h))

                    row_images.append([x, tile_w, tile])

                grid.tiles.append([y, tile_h, row_images])

            return grid

        def combine_grid(grid):
            def make_mask_image(r):
                r = r * 255 / grid.overlap
                r = r.astype(np.uint8)
                return Image.fromarray(r, "L")

            mask_w = make_mask_image(
                np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0)
            )
            mask_h = make_mask_image(
                np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1)
            )

            combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
            for y, h, row in grid.tiles:
                combined_row = Image.new("RGB", (grid.image_w, h))
                for x, w, tile in row:
                    if x == 0:
                        combined_row.paste(tile, (0, 0))
                        continue

                    combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
                    combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

                if y == 0:
                    combined_image.paste(combined_row, (0, 0))
                    continue

                combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
                combined_image.paste(
                    combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap)
                )

            return combined_image

        grid = split_grid(image=input_img, tile_w=512, tile_h=512, overlap=64)
        newtiles = []
        scale_factor = 1

        for y, h, row in grid.tiles:
            newrow = []
            for tiledata in row:
                x, w, tile = tiledata

                output = esrgan_enhance(model=model, img=tile)
                scale_factor = output.width // tile.width

                newrow.append([x * scale_factor, w * scale_factor, output])
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

        newgrid = Grid(
            newtiles,
            grid.tile_w * scale_factor,
            grid.tile_h * scale_factor,
            grid.image_w * scale_factor,
            grid.image_h * scale_factor,
            grid.overlap * scale_factor,
        )
        output = combine_grid(grid=newgrid)
        return output
