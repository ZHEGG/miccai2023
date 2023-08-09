import torch
import numpy as np
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler
from torch.utils.data import BatchSampler
import sys
try:
    from datasets.transforms import *
except:
    from transforms import *

import cv2
import os
import glob

class MultiPhaseLiverDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.size = args.img_size
        self.mode = args.mode
        self.is_training = is_training
        img_list = []
        lab_list = []
        phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase', 
                      'C-pre', 'C+A', 'C+V', 'C+Delay']
        if self.args.return_glb:
            glb_img_list = []
            phase_list_glb = ['T2WI_glb', 'DWI_glb', 'In Phase_glb', 'Out Phase_glb', 
                        'C-pre_glb', 'C+A_glb', 'C+V_glb', 'C+Delay_glb']

        if is_training:
            anno = np.loadtxt(args.train_anno_file, dtype=np.str_)
        else:
            anno = np.loadtxt(args.val_anno_file, dtype=np.str_)

        for item in anno:
            mp_img_list = []
            
            for phase in phase_list:
                mp_img_list.append(f'{args.data_dir}/{item[0]}/{phase}.nii.gz')
            if self.args.return_glb:
                mp_glb_img_list = []
                for phase in phase_list_glb:
                    mp_glb_img_list.append(f'{args.data_dir}/{item[0]}/{phase}.nii.gz')

            img_list.append(mp_img_list)
            if self.args.return_glb:
                glb_img_list.append(mp_glb_img_list)
            lab_list.append(item[1])

        self.img_list = img_list
        if self.args.return_glb:
            self.glb_img_list = glb_img_list
        self.lab_list = lab_list

    def get_labels(self,):
        return self.lab_list
    
    def __getitem__(self, index):
        patient_id = self.img_list[index][0].split('/')[-2]
        args = self.args
        label = int(self.lab_list[index])
        image = self.load_mp_images(self.img_list[index])
        if self.args.return_glb:
            glb_image = self.load_mp_images(self.glb_img_list[index])
        if self.is_training:
            if self.args.mixup and (label != 6):
                image = self.mixup(image,label)
                if self.args.return_glb:
                    glb_image = self.mixup(glb_image,label)
            image = self.transforms(image, args.train_transform_list)
            if self.args.return_glb:
                glb_image = self.transforms(glb_image,args.train_transform_list)
        else:
            image = self.transforms(image, args.val_transform_list)
            if self.args.return_glb:
                glb_image = self.transforms(glb_image,args.val_transform_list)
        image = image.copy()
        if self.args.return_glb:
            glb_image = glb_image.copy()
        
        if self.args.return_glb:
            return (image, label, glb_image, patient_id)
        else:
            return (image, label, patient_id)

    def mixup(self, image, label):
        # 类内mixup
        alpha = 1.0
        all_target_ind = [i for i, x in enumerate(self.lab_list) if int(x) == label]
        index = random.choice(all_target_ind)
        image_bal = self.load_mp_images(self.img_list[index])
        lam = np.random.beta(alpha, alpha)
        image = lam * image + (1 - lam) * image_bal

        return image

    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            # image = self.get_z_roi(image,16)
            image = resize3D(image, self.size, self.mode)
            image = image_normalization(image)
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def get_z_roi(self,image,z_num):
        Z,H,W = image.shape

        if Z > z_num:
            Z_mid = int(Z / 2)
            image_mid = image[Z_mid-int(z_num/2):Z_mid+int(z_num/2), :]
            return image_mid
        else:
            return image
    

    def transforms(self, mp_image, transform_list):
        args = self.args

        seed_diff = random.random()
        if seed_diff > 0.8:
            if 'diff_aug' in transform_list:
                diff_image = diffframe(mp_image)
                T,Z,H,W = mp_image.shape
                noise = np.random.normal(0,0.8,[T,Z,H,W]).astype(np.float32)
                diff_image*=noise
                mp_image = mp_image + diff_image

        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)

        if 'autoaugment' in transform_list:
            mp_image = image_net_autoaugment(mp_image)
            return mp_image

        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        
        seed = random.random()
        if seed > 0.9:
            if 'edge' in transform_list:
                mp_image = edge(mp_image)
        elif seed > 0.8:
            if 'emboss' in transform_list:
                mp_image = emboss(mp_image)
        elif seed > 0.4:
            if 'filter' in transform_list:
                seed2=random.random()
                if seed2>0.8:
                    mp_image = blur(mp_image)
                elif seed2>0.6:
                    mp_image = sharpen(mp_image)
                elif seed2>0.5:
                    mp_image = mask(mp_image)
            
        return mp_image

    def __len__(self):
        return len(self.img_list)

def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
        mode = 'instance',
):
    
    weights = get_sampling_probabilities(dataset,mode=mode)

    # np_lab_list = list(map(int,dataset.lab_list))
    # np_lab_list = np.array(np_lab_list)

    # weights_classes = [1.0/len(np.where(np_lab_list == i)[0]) for i in range(dataset.args.num_classes)]
    # weights = [weights_classes[i] for i in np_lab_list]

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            sampler = DistributedProxySampler(
                            ExhaustiveWeightedRandomSampler(weights, num_samples=len(dataset))
                        )
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
        if is_training:
            sampler = torch.utils.data.WeightedRandomSampler(weights=weights,num_samples = len(dataset),replacement=True)

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader

def get_sampling_probabilities(dataset,mode,ep=None, n_eps=None):
    np_lab_list = list(map(int,dataset.lab_list))
    class_count = np.unique(np_lab_list,return_counts=True)[1]
    sampling_probs = count_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[np_lab_list]

    return sample_weights

def count_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

if __name__ == "__main__":
    import yaml
    import parser
    import argparse
    from tqdm import tqdm

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='data/classification_dataset/images/', type=str)
    parser.add_argument(
        '--train_anno_file', default='data/classification_dataset/labels/train_fold1.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='data/classification_dataset/labels/val_fold1.txt', type=str)
    parser.add_argument('--train_transform_list', default=['random_crop',
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation',],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(14, 112, 112),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)

    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text
    
    args, args_text = _parse_args()
    args_text = yaml.load(args_text, Loader=yaml.FullLoader)
    args_text['img_size'] = 'xxx'
    print(args_text)

    args.distributed = False
    args.batch_size = 100

    dataset = MultiPhaseLiverDataset(args, is_training=True)
    data_loader = create_loader(dataset, batch_size=3, is_training=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    for images, labels in data_loader:
        print(images.shape)
        print(labels)

    # val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    # val_data_loader = create_loader(val_dataset, batch_size=10, is_training=False)
    # for images, labels in val_data_loader:
    #     print(images.shape)
    #     print(labels)