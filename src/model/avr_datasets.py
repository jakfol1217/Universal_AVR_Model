import copy
import glob
import os
import random

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
import torchvision
import ujson as json
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class Dsprites_OOO():
    def __init__(self, data_path, seed = None):
        if seed:
            random.seed(seed)
        with np.load(os.path.join(data_path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True, encoding='latin1') as file_dsprites:
            self.latent_sizes = file_dsprites['metadata'][()]['latents_sizes']
            self.latent_idxes = np.concatenate((self.latent_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
            self.dsprites = file_dsprites['imgs']
            
            
    def latent_dist_to_index(self, lat_dist):
        return np.dot(lat_dist, self.latent_idxes)
    
    
    def return_initial_dist(self):
        dist = [0 for i in range(len(self.latent_sizes))]
        for i, size in enumerate(self.latent_sizes):
            dist[i] = random.randint(0, size-1)
        return dist
    
    
    def return_new_dists(self, initial_dist, idxs):
        vals_1 = random.sample([i for i in range(self.latent_sizes[idxs[0]]) if i != initial_dist[idxs[0]]], k=3)
        vals_2 = [initial_dist[idxs[1]] for _ in range(2)] + random.sample([i for i in range(self.latent_sizes[idxs[1]]) if i != initial_dist[idxs[1]]], k=1)
        new_dists = []
        for val, val2 in zip(vals_1, vals_2):
            new_dist = copy.copy(initial_dist)
            new_dist[idxs[0]] = val
            new_dist[idxs[1]] = val2
            new_dists.append(new_dist)
        return new_dists
        
    
    def return_ooo(self, n):
        tasks = []
        targets = []
        latents = []
        for _ in range(n):
            new_task = self.return_single_task()
            tasks.append(new_task[0])
            targets.append(new_task[1])
            latents.append(new_task[2])
        
        return tasks, targets, latents
    
    def return_single_task(self):
        latent_types = sorted(random.sample(list(range(1, 6)), k=2), reverse=True)
        latent_dist = self.return_initial_dist()
        dists = [latent_dist, *self.return_new_dists(latent_dist, latent_types)]
        
        ooo_tasks = np.array([self.dsprites[self.latent_dist_to_index(dist)] for dist in dists])
        # random permutation of tasks
        
        task_idxes = list(range(4))
        random.shuffle(task_idxes)
        
        ooo_tasks = ooo_tasks[task_idxes]
        ooo_target = task_idxes.index(3)
        return ooo_tasks, ooo_target, latent_types
        
        
        


class HOIdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        annotation_path: str,
        dataset_type: str,
        img_size: int | None,
    ):

        self.data_files = [os.path.join(annotation_path, dataset_type)]
        if dataset_type:
            self.data_files = [f for f in self.data_files if dataset_type in f]
        self.file_sizes = []
        for file in self.data_files:
            with open(file) as f:
                file_hoi = json.load(f)
                self.file_sizes.append(len(file_hoi))
        
        self.idx_ranges = np.cumsum(self.file_sizes)
        
        if img_size:
            self.transforms = transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                    ]
            )
        else:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
            )
            
        self.data_path = data_path

    def __len__(self):
        return int(np.sum(self.file_sizes))
    
    def __getitem__(self, item):

        for i in range(len(self.idx_ranges)):
            if item < self.idx_ranges[i]:
                idx = i
                break
                
        file = self.data_files[idx]
        with open(file) as f:
            files_hoi = json.load(f)
            
        if idx == 0:
            file_hoi = files_hoi[item]
        else:
            file_hoi = files_hoi[item-self.idx_ranges[idx-1]]
        
        context = file_hoi[0] + file_hoi[1]
        context = [os.path.join(self.data_path, c['im_path']) for c in context]
        
        answers = []
        # adding random image as answer
        answers.append(context.pop(random.randint(0, 6)))
        answers.append(context.pop(random.randint(6, 12)))
        
        # why?
        # random answer flip
        if random.uniform(0,1) <= 0.5:
            target = np.asarray(1)
        else:
            target = np.asarray(0)
            answers = answers[::-1]
        
        images = context + answers
        img = [self.transforms(Image.open(im)) for im in images]
        img = torch.stack(img)
        
        return img, target


class LOGOdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        annotation_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        data_files = annotations[dataset_type]
        self.data_files = [os.path.join(data_path, file[:2], "images", file) for file in data_files]
        if img_size:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                    ]
            )
        else:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
            )

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, item):
        file = self.data_files[item]
        
        context = []
        answers = []
        ans_idx = random.randint(0,6)
        path_0 = os.path.join(file, "0")
        for i, file_0 in enumerate(os.listdir(path_0)):
            im = Image.open(os.path.join(path_0, file_0))
            if i == ans_idx:
                answers.append(self.transforms(im))
            else:
                context.append(self.transforms(im))

        ans_idx = random.randint(0,6)
        path_1 = os.path.join(file, "1")
        for i, file_1 in enumerate(os.listdir(path_1)): 
            im = Image.open(os.path.join(path_1, file_1))
            if i == ans_idx:
                answers.append(self.transforms(im))
            else:
                context.append(self.transforms(im))

        # random answer flip
        if random.uniform(0,1) <= 0.5:
            target = np.asarray(1)
        else:
            target = np.asarray(0)
            answers = answers[::-1]
        
        images = context + answers
        img = torch.stack(images)
        
        return img, target
        
        

    
class DEEPIQdataset(Dataset):
    def __init__(self, data_path, img_size=None):
        self.data    = glob.glob(os.path.join(data_path, "*.png"))
        self.answers = pd.read_csv(glob.glob(os.path.join(data_path, "*.csv"))[0], header=None)
        if img_size:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                    ]
            )
        else:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
            )
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, item):
        img_path = self.data[item]
        img_split = self.split_ooo_images(Image.open(img_path))
        img = [self.transforms(Image.fromarray(split.astype(np.uint8))) for split in img_split]
        img = torch.stack(img)
                                   
        target = np.asarray(self.answers[0][item])
        
        return img, target
        
    
    def split_ooo_images(self, image):
        images = []
        for window in range(0, image.size[0], 100):
            images.append(np.array(image)[:,window:window+100])
        return images

    
class DOPTdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_file = os.path.join(data_path, dataset_type)
        self.file = np.load(self.data_file, mmap_mode='r')
        self.file_size = len(self.file)
        if img_size:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

    def __len__(self):
        return self.file_size

    def __getitem__(self, item):
        cur_file = self.file[item]
        # FIXME: cur_file is in 20x64x64 shape so I guess we can choose how to shape the task, now 0-16 images, 16-20 answers
        #        can possible create many tasks from these 20 images (different starting point/range between images).
        images = cur_file[:-4]
        answers = cur_file[-4:]
        idx = list(range(len(answers)))
        random.shuffle(idx)
        answers = answers[idx]

        images = np.concatenate([images, answers])
        img = torch.stack([self.transforms(Image.fromarray((im*255).astype(np.uint8))) for im in images])

        target = np.asarray(idx.index(0))

        return img, target


class DSPRITESdataset(Dataset):
    def __init__(self, data_path, img_size=None, num_tasks=1000):
        
        self.DSPRITE_tasker = Dsprites_OOO(data_path, 12)
        self.tasks, self.targets, _ = self.DSPRITE_tasker.return_ooo(num_tasks)
        
        
        if img_size:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                    ]
            )
        else:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
            )
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, item):
        tasks = self.tasks[item]
        
        img = torch.stack([self.transforms(Image.fromarray(im.astype(np.uint8))) for im in tasks])
        
        target = np.asarray(self.targets[item])
        
        return img, target
        
        
    
class IRAVENdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        regimes: list[str],
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = []
        for regime in regimes:
            files = [f for f in glob.glob(os.path.join(data_path, regime, "*.npz")) if dataset_type in f]
            self.data_files += files
        if img_size:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                    ]
            )
        else:
            self.transforms = transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
            )
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, item):
        data = np.load(self.data_files[item], mmap_mode='r')
        images = data['image']

        target = np.asarray(data['targ'])
        
        img = torch.stack([self.transforms(Image.fromarray(im.astype(np.uint8))) for im in images])
        
        return img, target


class MNSdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = glob.glob(os.path.join(os.path.join(data_path, dataset_type), "*.npz"))
        if img_size:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, item):
        data = np.load(self.data_files[item], mmap_mode='r')
        images = data['image']
        target = np.asarray(data['target'])
        
        img = torch.stack([self.transforms(Image.fromarray(im.astype(np.uint8))) for im in images])
        
        return img, target

    
    
class PGMdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        regimes: list[str],
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = []
        for regime in regimes:
            files = [f for f in glob.glob(os.path.join(data_path, regime, "*.npz")) if dataset_type in f]
            self.data_files += files

        if img_size:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
            

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, item):
        data = np.load(self.data_files[item], mmap_mode='r')
        images = data['image']
        images = images.reshape(-1,160,160)
        target = np.asarray(data['target'])
        
        img = torch.stack([self.transforms(Image.fromarray(im.astype(np.uint8))) for im in images])
        
        return img, target


class VAECdataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_file = os.path.join(data_path, dataset_type)
        with h5py.File(self.data_file) as f:
            self.file_size = len(f)

        if img_size:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

        self.data_path = data_path

    def __len__(self):
        return self.file_size

    def __getitem__(self, item):

        file = self.data_file

        local_item = str(item)
        
        with h5py.File(file) as f:
            context = [f[local_item]['imgs'][_idx] for _idx in list(f[local_item]['ABCD'])[:3]]
            idx_hy = [i for i in list(f[local_item]['not_D']) if i not in list(f[local_item]['ABCD'])][:3] + [list(f[local_item]['ABCD'])[3]]
            random.shuffle(idx_hy)
            answers = np.asarray(f[local_item]['imgs'])[idx_hy]
            target = np.asarray(idx_hy.index(list(f[local_item]['ABCD'])[3]))

        images = np.concatenate([context, answers])
        img = torch.stack([self.transforms(Image.fromarray(im.astype(np.uint8))) for im in images])

        return img, target


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def _test(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg.dataset)

    # Example of usage
    train_dataset = HOIdataset(cfg.dataset.tasks.bongard_hoi.train)
    img, target = train_dataset[0]
    print(img.shape)  # [14, 3, 256, 256]
    print(target)  # 0

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    val_dataset_logo = LOGOdataset(cfg.dataset.tasks.bongard_logo.val)
    img, target = val_dataset_logo[0]
    print(img.shape)  # torch.Size([14, 3, 512, 512])
    print(target)  # 1

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    test_dataset_dopt = DOPTdataset(cfg.dataset.tasks.dopt.test)
    img, target = test_dataset_dopt[0]
    print(img.shape)  # torch.Size([20, 1, 64, 64])
    # BUG: This dataset is not working, it's not returning the correct images
    print(target)  # 1

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    test_dataset_vaec = VAECdataset(cfg.dataset.tasks.vaec.test)
    img, target = test_dataset_vaec[0]
    print(img.shape)  # torch.Size([8, 3, 128, 128])
    print(target)  # 3

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    test_dataset_iraven = IRAVENdataset(cfg.dataset.tasks.iraven.test)
    img, target = test_dataset_iraven[0]
    print(img.shape)  # torch.Size([16, 1, 160, 160])
    print(target)  # 7

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    train_dataset_mns = MNSdataset(cfg.dataset.tasks.mns.train)
    img, target = train_dataset_mns[0]
    print(img.shape)  # torch.Size([3, 1, 160, 160])
    print(target)  # 4

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")

    train_dataset_pgm = PGMdataset(cfg.dataset.tasks.pgm.train)
    img, target = train_dataset_pgm[0]
    print(img.shape)  # torch.Size([16, 1, 160, 160])
    print(target)  # 6

    # for i in range(img.shape[0]):
    #     img_pil = transforms.ToPILImage()(img[i])
    #     img_pil.save(f"img_{i}.png")
    # TODO: Add config classes to make it easier to read and know what can be added https://hydra.cc/docs/tutorials/structured_config/schema/


if __name__ == "__main__":
    _test()
