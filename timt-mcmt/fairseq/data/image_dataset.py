import os
import torch
import random
class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, split: str):
        self.split = split
        self.img_feat_path = []
        for i in range(len(os.listdir(feat_path))):
            self.img_feat_path.append(os.path.join(feat_path,split+str(i)+".pth"))
        self.size = len(self.img_feat_path)


    def __getitem__(self, idx):
        return torch.load(self.img_feat_path[idx])

    def __len__(self):
        return self.size

# load to memory
# class ImageDataset(torch.utils.data.Dataset):
#     """
#     For loading image datasets
#     """
#     def __init__(self, feat_path: str, split: str):
#         self.split = split
#         for i in range(len(os.listdir(feat_path))):
#             if i == 0:
#                 self.img_feat = torch.load(os.path.join(feat_path,split+str(i)+".pth"))
#             else:
#                 self.img_feat = torch.cat((self.img_feat,torch.load(os.path.join(feat_path,split+str(i)+".pth"))))
#         self.size = self.img_feat.shape[0]

#     def __getitem__(self, idx):
#         return self.img_feat[idx]
#         # return self.img_feat[idx]

#     def __len__(self):
#         return self.size

