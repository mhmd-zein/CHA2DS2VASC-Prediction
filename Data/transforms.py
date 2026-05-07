import numpy as np
import torch
from PIL import Image
from monai.transforms import MapTransform
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    Randomizable,
    Rotate90d,
    ToNumpyd,
    ToTensord,
    RepeatChanneld,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotated,
    RandAdjustContrastd,
    RandAdjustContrast,
    RandGaussianSmoothd,
    RandGaussianNoised,
    Resized,

    Flipd
    )
import torch
import numpy as np
from PIL import Image
from monai.transforms import MapTransform



class Load_ch_d(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def load_stack_ch(self, image_paths: dict):
        images = []
        for layer, img_path in image_paths.items():
            image = Image.open(img_path).convert("L")
            image = image.resize((256, 256))
            image_array = np.array(image, dtype=np.float32)
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            images.append(image_tensor)

        stacked_image = torch.cat(images, dim=0)
        return stacked_image

    def __call__(self, data):
        for key in self.keys:
            if key not in data:
                raise KeyError(f"Key {key} not found in data.")
            image_paths = data[key]
            if not isinstance(image_paths, (list, dict)):
                raise TypeError(f"Expected a list of image paths for key '{key}', got {type(image_paths)}.")
            stacked_image = self.load_stack_ch(image_paths)
            output_key = self.output_key or key
            data[output_key] = stacked_image
        return data
    


class Load_Replicate_Channeld(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def load_and_replicate(self, image_paths):
        for key in image_paths:
            image_path = image_paths[key]
            image = Image.open(image_path).convert("L")
            image = image.resize((256, 256))
            image_array = np.array(image, dtype=np.float32)
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            replicated_image = image_tensor.repeat(3, 1, 1)
            if key == 'perf_map':
                break 
        return replicated_image

    def __call__(self, data):
        for key in self.keys:
            if key not in data:
                raise KeyError(f"Key {key} not found in data.")
            image_paths = data[key]
            if not isinstance(image_paths, (list, dict)):
                raise TypeError(f"Expected a list of image paths for key '{key}', got {type(image_paths)}.")
            replicated_image = self.load_and_replicate(image_paths)
            output_key = self.output_key or key
            data[output_key] = replicated_image
        return data


train_transforms_1_channel = Compose([
    Load_Replicate_Channeld(keys=['image_paths'], output_key='image'),
    NormalizeIntensityd(keys=['image']),
    # Resized(keys=['image'], spatial_size=(512, 512), mode='bilinear'),
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
    RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), prob=0.3),
    RandGaussianNoised(keys=['image'], mean=0.0, std=0.1, prob=0.3),
    RandAdjustContrastd(keys=['image'], gamma=(0.8, 1.2), prob=0.3),
    # RandRotated(keys=['image'], range_x=(-15, 15), range_y=(-15, 15), prob=0.5, mode='bilinear'),
    ToTensord(keys=['image']),
    ToTensord(keys=['label'])

])

test_transforms_1_channel = Compose([
    Load_Replicate_Channeld(keys=['image_paths'], output_key='image'),
    NormalizeIntensityd(keys=['image']),
    # Resized(keys=['image'], spatial_size=(512, 512), mode='bilinear'),
    ToTensord(keys=['image']),
    ToTensord(keys=['label'])
])

train_transforms_multi_channel = Compose([
    Load_ch_d(keys=['image_paths'], output_key='image'),
    NormalizeIntensityd(keys=['image']),
    # Resized(keys=['image'], spatial_size=(512, 512), mode='bilinear'),
    # RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
    RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), prob=0.3),
    RandGaussianNoised(keys=['image'], mean=0.0, std=0.1, prob=0.3),
    RandAdjustContrastd(keys=['image'], gamma=(0.8, 1.2), prob=0.3),
    # RandRotated(keys=['image'], range_x=(-15, 15), range_y=(-15, 15), prob=0.5, mode='bilinear'),
    ToTensord(keys=['image']),
    ToTensord(keys=['label'])
    
])

test_transforms_multi_channel = Compose([
    Load_ch_d(keys=['image_paths'], output_key='image'),
    NormalizeIntensityd(keys=['image']),
    # Resized(keys=['image'], spatial_size=(512, 512), mode='bilinear'),
    ToTensord(keys=['image']),
    ToTensord(keys=['label'])
])