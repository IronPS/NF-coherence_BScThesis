import torch
import numpy as np
from nflows.transforms.base import MultiscaleCompositeTransform

class MultiscaleCompositeTransformAux(MultiscaleCompositeTransform):
    def get_split_indices(self):
        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)
        
        return split_indices
