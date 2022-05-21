import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
from natsort import natsort_keygen

class CelebADataset(Dataset):
    def __init__(self, data_path, partition, transform=None, file_extension=".pt", size=None, masks=None, mirrored=False):
        """
        
        -----------
        Parameters:
            data_path:
                Path to the dataset
                It must contain two folders: images and eval.
                The eval folder must contain the text file "list_eval_partition.txt" 
                listing the train, test and eval partitions.
            partition:
                A string: "train", "test" or "eval"
            transform:
                default: None
                A function that will transform the output vector corresponding to the image
                Must be able to receive an input tensor of corresponding dimensionality (B, C, W, H)
            file_extension:
                The extension used by the data files
            size:
                The number of datapoints to use
            masks:
                A dictionary with pairs of feature name (e.g., "Black_Hair") and value: 1 for present -1 for not present
        """
        self.data_path = data_path
        self.images_path = self.data_path + "/images"
        self.transform = transform
        self.file_extension = file_extension
        self.size = size
        self.masks = masks

        self.data = None
        self.attributes = None

        self._mirror_image = mirrored
        self._getData(partition)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        dpoint = self.data.iloc[index]
        fpath = "{}/{}".format(self.images_path, dpoint.image)
        if self.file_extension == ".jpg" \
           or self.file_extension == ".png":
            X = Image.open(fpath).convert('RGB')

        elif self.file_extension == ".pt":
            X = torch.load(fpath)

        if self._mirror_image and dpoint.mirrored:
            X = ImageOps.mirror(X)

        if self.transform:
            X = self.transform(X)


        return X, int('.'.join(dpoint.image.split('.')[:-1]))

    def get_attributes(self, index):
        return self.data.iloc[index].values

    def _getData(self, partition):
        def partitionID(name):
            if name == "train":
                return 0
            elif name == "test":
                return 1
            else:
                return 2
        
        if self.file_extension == ".jpg" \
           or self.file_extension == ".png":
           self._mirror_image = True
        #    self._mirror_image = False # do nothing

        transform = None
        if self.file_extension != ".jpg":
            def replace_fn(x):
                return x.replace(".jpg", self.file_extension)
            transform = np.vectorize(replace_fn)
        
        ids = pd.read_csv(self.data_path + "/metadata/list_eval_partition.txt", names=["image", "partition"], header=None, delimiter=' ') \
               .groupby(by="partition")["image"] \
               .get_group(partitionID(partition))

        if not self.size:
            self.size = ids.shape[0]

        if self._mirror_image:
            ids = ids.head(self.size//2)
        else:
            ids = ids.head(self.size)


        df_attrs = pd.read_csv(self.data_path + "/metadata/list_attr_celeba.txt", header=1, delim_whitespace=True)
        df_attrs = df_attrs.rename_axis('image')
        
        if transform:
            ids = transform(ids)
            df_attrs = df_attrs.reset_index()
            df_attrs.image = transform(df_attrs.image)
            df_attrs = df_attrs.set_index('image')

        if self.masks:
            for m, v in self.masks.items():
                df_attrs = df_attrs.loc[df_attrs[m] == v]

        ids = pd.DataFrame({'image': ids}).set_index('image')
        self.data = df_attrs.join(ids, how='inner', on='image').reset_index()
        if self._mirror_image:
            self.data['mirrored'] = True
            self.data = pd.concat([self.data]*2, ignore_index=False).sort_index(ignore_index=True)
            self.data.loc[1::2, 'mirrored'] = False

        self.data.sort_values(
            by='image', 
            key=natsort_keygen(),
            inplace=True
        )

        self.attributes = dict(zip(self.data.columns.values, range(self.data.columns.shape[0])))
