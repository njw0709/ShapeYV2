import torchvision.datasets as datasets
from torch.utils.data import Dataset
from itertools import combinations
import math
import psutil
import numpy as np
from typing import List

class CombinationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.comb = list(combinations(dataset, 2))

    def __getitem__(self, index):
        img1, img2 = self.comb[index]
        return img1, img2

    def __len__(self):
        return len(self.comb)

    def cut_dataset(self, index):
        self.comb = self.comb[index:]


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class FeatureTensorDatasetWithImgName(Dataset):
    def __init__(self, feature_tensor, img_name_array):
        self.feature_tensor = feature_tensor
        self.imgnames = img_name_array

    def __getitem__(self, index):
        feat = self.feature_tensor[index, :]
        imgname = self.imgnames[index]
        return imgname, feat

    def __len__(self):
        return len(self.imgnames)


class PermutationIndexDataset(Dataset):
    def __init__(self, datalen):
        self.datalen = datalen

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        return idx1, idx2


class OriginalandPostProcessedPairsDataset(Dataset):
    def __init__(self, original_feat_dataset, postprocessed_feat_dataset):
        self.original = original_feat_dataset
        self.postprocessed = postprocessed_feat_dataset
        self.datalen = len(self.postprocessed)

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        s1 = self.original[idx1]
        s2 = self.postprocessed[idx2]
        return (idx1, s1), (idx2, s2)

    def __len__(self):
        return len(self.original) ** 2


class PermutationPairsDataset(Dataset):
    def __init__(self, original_feat_dataset, postprocessed=None):
        self.original = original_feat_dataset
        self.datalen = len(self.original)
        self.postprocessed = postprocessed

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        s1 = self.original[idx1]
        if self.postprocessed is not None:
            s2 = self.postprocessed[idx2]
        else:
            s2 = self.original[idx2]
        return (idx1, s1), (idx2, s2)

    def __len__(self):
        return len(self.original) ** 2

class PermutationPairsSameObj(PermutationPairsDataset):
    def __init__(self, original_feat_dataset, numviews_sameobj=341,postprocessed=None):
        super().__init__(original_feat_dataset, postprocessed)
        self.datalen = len(self.original)
        self.numviews_sameobj = numviews_sameobj
    
    def __getitem__(self, index):
        objidx = index // self.numviews_sameobj**2
        viewidx = index % self.numviews_sameobj**2
        idx1 = self.numviews_sameobj*objidx + viewidx//self.numviews_sameobj
        idx2 = self.numviews_sameobj*objidx + viewidx%self.numviews_sameobj
        s1 = self.original[idx1]
        if self.postprocessed is not None:
            s2 = self.postprocessed[idx2]
        else:
            s2 = self.original[idx2]
        return (idx1, s1), (idx2, s2)
    
    def __len__(self):
        numobj = self.datalen // self.numviews_sameobj
        return numobj * self.numviews_sameobj**2

class PermutationDatasetWithNNBatches(PermutationPairsDataset):
    def __init__(self, original_feat_dataset, batch_idxs: List[np.ndarray], postprocessed=None, ):
        super().__init__(original_feat_dataset, postprocessed)
        self.batch_idxs = batch_idxs
        self.batch_dims = [b.size for b in self.batch_idxs]
        if len(set(self.batch_dims[:-1])) == 1:
            self.same_batch_size = True
        else:
            self.same_batch_size = False
        batch_lengths = np.array([dim**2 for dim in self.batch_dims])
        self.batch_index_cutoff = np.cumsum(batch_lengths)

    def __getitem__(self, index):
        # compute which batch you are in
        if self.same_batch_size:
            batch_id = int(index/(self.batch_dims[0])**2)
            within_batch_idx = index - batch_id*(self.batch_dims[0]**2)
        else:
            batch_id = self.get_batch_num(index)
            within_batch_idx = index - self.batch_index_cutoff[batch_id - 1]
        
        bidx1 = int(within_batch_idx/self.batch_dims[batch_id])
        bidx2 = within_batch_idx % self.batch_dims[batch_id]
        imgidx1 = self.batch_idxs[batch_id][bidx1]
        imgidx2 = self.batch_idxs[batch_id][bidx2]
        s1 = self.original[imgidx1]
        if self.postprocessed is not None:
            s2 = self.postprocessed[imgidx2]
        else:
            s2 = self.original[imgidx2]
        return (imgidx1, s1), (imgidx2, s2)
    
    def __len__(self):
        return self.batch_index_cutoff[-1]
        
    def get_batch_num(self, index):
        batch_id = 0
        for l in self.batch_lengths:
            index -= l
            if index < 0:
                return batch_id
            else:
                batch_id +=1
    

class HDFDataset(Dataset):
    def __init__(self, hdfstore, mem_usage=0.85):
        self.hdfstore = hdfstore
        self.datalen = len(self.hdfstore)
        self.pull_data_to_cache(mem_usage)
        if not self.all_in_cache:
            print("initializing placeholder cache list")
            self.cache_length = int(
                psutil.virtual_memory().available * 0.85 / self.hdfstore[0].nbytes
            )
            self.in_cache_idx = [None] * self.cache_length
            self.in_cache = [None] * self.cache_length
            self.cache_counter = 0

    def __getitem__(self, index):
        if not self.all_in_cache:
            if index in self.in_cache_idx:
                return self.in_cache[self.in_cache_idx.index(index)]
            else:
                self.in_cache_idx[self.cache_counter] = index
                data = self.hdfstore[index]
                self.in_cache[self.cache_counter] = data
                self.cache_counter += 1
                self.cache_counter %= self.cache_length
                return data
        return self.hdfstore[index]

    def __len__(self):
        return self.datalen

    def pull_data_to_cache(self, mem_usage):
        single_row = self.hdfstore[0]
        if (
            psutil.virtual_memory().available * mem_usage
            < single_row.nbytes * self.datalen
        ):
            print("Not enough memory to pull data to cache")
            self.all_in_cache = False
        else:
            print("Pulling data to cache")
            self.hdfstore = self.hdfstore[:]
            self.all_in_cache = True
            print("Done pulling data to cache")
