"""Unmodified v3 preprocessing function, for raw-source tooling only."""
import io
import numpy as np
import PIL
from PIL import Image, ImageOps

def preprocess(raw):
    with Image.open(io.BytesIO(raw)) as source:
        original_size = list(source.size)
        orientation = source.getexif().get(274)
        image = ImageOps.exif_transpose(source).convert('RGB')
        oriented_size = list(image.size)
        w, h = image.size
        if (w, h) == (256, 256):
            resized = (w, h)
            box = [0, 0, w, h]
            resized_applied = crop_applied = False
        else:
            short = min(w, h)
            resized = (w * 256 // short, h * 256 // short)
            image = image.resize(resized, Image.Resampling.BICUBIC)
            left, top = (resized[0] - 256) // 2, (resized[1] - 256) // 2
            box = [left, top, left + 256, top + 256]
            image = image.crop(tuple(box))
            resized_applied = crop_applied = True
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / np.float32(255.)
    return array, dict(original_size_wh=original_size, exif_orientation=orientation,
                      exif_transpose=True, oriented_size_wh=oriented_size, convert_mode='RGB',
                      resize_size_wh=list(resized), resize_applied=resized_applied,
                      interpolation='Pillow BICUBIC', long_dimension_rounding='floor',
                      crop_box_xyxy=box, crop_applied=crop_applied, random_augmentation=False,
                      pillow_version=PIL.__version__, dtype='float32', layout='CHW', scale='uint8 / 255.0')
