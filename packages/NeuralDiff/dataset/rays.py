import json
import os
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

from .utils import *


def load_meta(root, name="meta.json", N_vocab=1000):
    """Load meta information per scene and frame (nears, fars, poses etc.)."""
    path = os.path.join(root, name)
    with open(path, "r") as fp:
        ds = json.load(fp)
    for k in ["nears", "fars", "images", "poses"]:
        ds[k] = {int(i): ds[k][i] for i in ds[k]}
        if k == "poses":
            ds[k] = {i: np.array(ds[k][i]) for i in ds[k]}
    ds["intrinsics"] = np.array(ds["intrinsics"])
    print('ds.keys(): ', ds.keys())
    # ds.keys():  dict_keys(['ids_all', 'images', 'intrinsics', 'camera', 'image_w', 'image_h', 'nears', 'fars', 'poses', 'ids_train', 'ids_val', 'ids_test'])
    ds_copy = copy.deepcopy(ds)
    for ds_k in ds_copy.keys():
        if ds_k in ['camera', 'intrinsics', 'image_w', 'image_h', ]:
            continue
        if isinstance(ds_copy[ds_k], dict):
            print('{} is a dict, N_vocab: {}'.format(ds_k, N_vocab))
            for k, v in ds_copy[ds_k].items():
                if int(k) >= N_vocab:
                    ds[ds_k].pop(k)

        if isinstance(ds_copy[ds_k], list):
            print('{} is a list, N_vocab: {}'.format(ds_k, N_vocab))
            ds_list = list()
            for i, elem in enumerate(ds_copy[ds_k]):
                if int(elem) < N_vocab:
                    ds_list.append(elem)
            ds[ds_k] = ds_list
    return ds


class EPICDiff(Dataset):
    def __init__(self, vid, root="data/EPIC-Diff", split=None, N_vocab=1000):

        self.root = os.path.join(root, vid)
        self.N_vocab = N_vocab
        self.vid = vid
        self.img_w = None  # 154  # 228
        self.img_h = None  # 175  # 128
        self.split = split
        self.val_num = 1
        self.transform = torchvision.transforms.ToTensor()
        self.init_meta()

    def imshow(self, index):
        plt.imshow(self.imread(index))
        plt.axis("off")
        plt.show()

    def imread(self, index):
        return plt.imread(os.path.join(self.root, "frames", self.image_paths[index]))

    def x2im(self, x, type_="np"):
        """Convert numpy or torch tensor to numpy or torch 'image'."""
        w = self.img_w
        h = self.img_h
        # print(w, h, x.shape)  # 228 128 torch.Size([26950, 3])
        if len(x.shape) == 2 and x.shape[1] == 3:
            x = x.reshape(h, w, 3)
        else:
            x = x.reshape(h, w)
        if type(x) == torch.Tensor:
            x = x.detach().cpu()
            if type_ == "np":
                x = x.numpy()
        elif type(x) == np.array:
            if type_ == "pt":
                x = torch.from_numpy(x)
        return x

    def rays_per_image(self, idx, pose=None):
        """Return sample with rays, frame index etc."""
        sample = {}
        if pose is None:
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[idx])
        else:
            sample["c2w"] = c2w = pose

        sample["im_path"] = self.image_paths[idx]

        img = Image.open(os.path.join(self.root, "frames", self.image_paths[idx]))
        img_w, img_h = img.size
        if self.img_h is None:
            self.img_h = img_h
            print('img_h: {}'.format(self.img_h))
        if self.img_w is None:
            self.img_w = img_w
            print('img_w: {}'.format(self.img_w))
        img = self.transform(img)  # (3, h, w)
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

        directions = get_ray_directions(img_h, img_w, self.K)
        rays_o, rays_d = get_rays(directions, c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(directions, c2c)

        rays_t = idx * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays
        sample["img_wh"] = torch.LongTensor([img_w, img_h])
        sample["ts"] = rays_t
        sample["rgbs"] = img

        return sample

    def init_meta(self):
        """Load meta information, e.g. intrinsics, train, test, val split etc."""
        meta = load_meta(self.root, N_vocab=self.N_vocab)
        self.img_ids = meta["ids_all"]
        self.img_ids_train = meta["ids_train"]
        self.img_ids_test = meta["ids_test"]
        self.img_ids_val = meta["ids_val"]
        self.poses_dict = meta["poses"]
        self.nears = meta["nears"]
        self.fars = meta["fars"]
        self.image_paths = meta["images"]
        self.K = meta["intrinsics"]

        if self.split == "train":
            # create buffer of all rays and rgb data
            self.rays = []
            self.rgbs = []
            self.ts = []

            for idx in self.img_ids_train:
                sample = self.rays_per_image(idx)
                self.rgbs += [sample["rgbs"]]
                self.rays += [sample["rays"]]
                self.ts += [sample["ts"]]

            self.rays = torch.cat(self.rays, 0)  # ((N_images-1)*h*w, 8)
            self.rgbs = torch.cat(self.rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.ts = torch.cat(self.ts, 0)

    def __len__(self):
        if self.split == "train":
            # rays are stored concatenated
            return len(self.rays)
        if self.split == "val":
            # evaluate only one image, sampled from val img ids
            return 1
        else:
            # choose any image index
            return max(self.img_ids)

    def __getitem__(self, idx, pose=None):

        if self.split == "train":
            # samples selected from prefetched train data
            sample = {
                "rays": self.rays[idx],
                "ts": self.ts[idx, 0].long(),
                "rgbs": self.rgbs[idx],
            }

        elif self.split == "val":
            # for tuning hyperparameters, tensorboard samples
            idx = random.choice(self.img_ids_val)
            sample = self.rays_per_image(idx, pose)

        elif self.split == "test":
            # evaluating according to table in paper, chosen index must be in test ids
            assert idx in self.img_ids_test
            sample = self.rays_per_image(idx, pose)

        else:
            # for arbitrary samples, e.g. summary video when rendering over all images
            sample = self.rays_per_image(idx, pose)

        return sample
