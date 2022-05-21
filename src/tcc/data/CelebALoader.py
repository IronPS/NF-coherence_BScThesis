
from tcc.data import CelebADataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch
from torchvision import transforms

from tcc.transforms.PillowTransforms import CenterCropResize

def _celebaImagePreprocessTransform(crop_size=148, new_size=[64, 64]):
    return transforms.Compose([
            CenterCropResize(crop_size=crop_size, new_size=new_size),
            transforms.PILToTensor()
        ])

def _collate_fn(batch):
    indices, data = [], []
    for (data_, idx_) in batch:
        data.append(data_)
        indices.append(idx_)
            
    data = torch.stack(data)
    indices = torch.tensor(indices, dtype=type(indices[-1]))

    return data, indices

class CelebALoader(DataLoader):
    def __init__(self, params, type="train", **kwargs):

        transform = None
        if params.transform == "celebaImagePreprocessTransform":
            transform = _celebaImagePreprocessTransform(params.crop_size, [params.to_image_size, params.to_image_size])
        elif params.transform == "celeba64ImagePreprocessTransform":
            transform = _celebaImagePreprocessTransform(148, [64, 64])

        self._dataset = CelebADataset(params.dataset_path, type, transform=transform, file_extension=params.file_extension, size=getattr(params, 'max_samples', None), masks=getattr(params, 'masks', None), mirrored=getattr(params, 'mirror_data', False))
        super().__init__(self._dataset, collate_fn=_collate_fn, **kwargs)

    