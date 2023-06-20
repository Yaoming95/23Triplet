import os
import torch
import random

class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, mask_path: str, shuffle_img=False):
        self.img_feat = torch.load(feat_path)
        self.img_feat_mask = None
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)
        self.img_feat = [torch.tensor(x) for x in self.img_feat]
        if shuffle_img:
            random.shuffle(self.img_feat)
        self.size = len(self.img_feat)

    def __getitem__(self, idx):
        if self.img_feat_mask is None:
            return self.img_feat[idx], None
        else:
            return self.img_feat[idx], self.img_feat_mask[idx]

    def __len__(self):
        return self.size
