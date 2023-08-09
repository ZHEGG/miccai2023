#!/usr/bin/env python3
'''
generate prediction on unlabeled data
'''
import argparse
import os
import json
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from contextlib import suppress
from torch.utils.data.dataloader import DataLoader
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import setup_default_logging, set_jit_legacy
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
from PIL import Image
import cv2

import models
from metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='Prediction on unlabeled data')

parser.add_argument('--img_size', default=(16, 128, 128),
                    type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(14, 112, 112),
                    type=int, nargs='+', help='cropped image size.')
parser.add_argument('--data_dir', default='./images/', type=str)
parser.add_argument('--val_anno_file', default='./labels/test.txt', type=str)
parser.add_argument('--val_transform_list',
                    default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=7,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--team_name', default='', type=str,
                    required=True, help='Please enter your team name')
parser.add_argument('--mode', type=str, default='trilinear', help='interpolate mode (trilinear, tricubic)')
parser.add_argument('--return_visualization', action='store_true', default=False, help='if return_visualization')
parser.add_argument('--return_hidden', action='store_true', default=False, help='if return_model_hidden')
parser.add_argument('--return_glb', action='store_true', default=False, help='if return_glb_input')

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    # if args.native_amp:
    #     amp_autocast = torch.cuda.amp.autocast
    #     _logger.info('Validating in mixed precision with native PyTorch AMP.')
    # elif args.apex_amp:
    #     _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    # else:
    #     _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        pretrained_cfg=None,
        return_visualization=args.return_visualization,
        return_hidden = args.return_hidden)

    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' %
                 (args.model, param_count))

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.num_gpu)))

    dataset = MultiPhaseLiverDataset(args, is_training=False)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=args.pin_mem,
                        shuffle=False)

    predictions = []
    labels = []
    patient_ids = []
    embedding_list_test = []

    model.eval()
    # cam_extractor = SmoothGradCAMpp(model,target_layer=model.blocks2[-1].mlp,input_shape=(8,14,112,112))

    pbar = tqdm(total=len(dataset))
    with torch.no_grad():
        for batch_idx, item in enumerate(loader):
            if args.return_glb:
                input = item[0]
                target = item[1]
                global_input = item[2]
                patient_id = item[-1]
                input, target, global_input = input.cuda(), target.cuda(), global_input.cuda()
            else:
                input = item[0]
                target = item[1]
                patient_id = item[-1]
                input, target = input.cuda(), target.cuda()
            patient_id = list(patient_id)
            # compute output
            with amp_autocast():
                if args.model == "uniformer_small_original" or args.model == "uniformer_base_original" or args.model == "uniformer_xs_original" or args.model == "uniformer_xxs_original" :
                    output,visualizations = model(input)
                else:
                    raise ValueError('invalid model input')

                ####################activate_map####################
                # for img_idx in range(input.shape[0]):
                #     activate_map = cam_extractor(target[img_idx].item(),output[img_idx]) # N,Z,H,W
                #     for modal in range(input.shape[1]):
                #         cur_img = input[img_idx][modal] # Z,H,W
                #         for z in range(input.shape[2]):
                #             cur_img_z = cur_img[z]
                #             cur_img_z = cur_img_z.cpu().numpy()
                #             cur_img_z = np.repeat(cur_img_z[:, :, np.newaxis], 3, axis=-1)
                #             cur_img_z = (cur_img_z-cur_img_z.min())/(cur_img_z.max()-cur_img_z.min())
                #             cur_img_z = (cur_img_z * 255).astype(np.uint8)
                #             mask = to_pil_image(activate_map[0][img_idx][z], mode='F')
                #             cmap = cm.get_cmap('jet')
                #             overlay = mask.resize(cur_img_z.shape[:2], resample=Image.BICUBIC)
                #             overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8).transpose(1,0,2)[:,:,::-1]

                #             out_img = cur_img_z*0.5+overlay*0.5

                #             concat_img = np.concatenate([cur_img_z,overlay,out_img],axis=1)
                #             cv2.imwrite('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/vis/vis_att_%s_%s.jpg'%(img_idx,z),concat_img)

                ####################attention####################
                # multi_head_attn_map = visualizations[0] #N,head_num,Z,H,W
                # for img_idx in range(input.shape[0]):
                #     for modal in range(input.shape[1]):
                #         cur_img = input[img_idx][modal] # Z,H,W
                #         for head in range(multi_head_attn_map.shape[1]):
                #             cur_head_attn_map = multi_head_attn_map[img_idx][head] # Z,H,W
                #             for z in range(input.shape[2]):
                #                 cur_img_z = cur_img[z]
                #                 cur_img_z = cur_img_z.cpu().numpy()
                #                 cur_img_z = np.repeat(cur_img_z[:, :, np.newaxis], 3, axis=-1)
                #                 cur_img_z = (cur_img_z-cur_img_z.min())/(cur_img_z.max()-cur_img_z.min())
                #                 cur_img_z = (cur_img_z * 255).astype(np.uint8)
                #                 mask = to_pil_image(cur_head_attn_map[z], mode='F')
                #                 cmap = cm.get_cmap('jet')
                #                 overlay = mask.resize(cur_img_z.shape[:2], resample=Image.BICUBIC)
                #                 overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8).transpose(1,0,2)[:,:,::-1]

                #                 out_img = cur_img_z*0.5+overlay*0.5

                #                 concat_img = np.concatenate([cur_img_z,overlay,out_img],axis=1)
                #                 cv2.imwrite('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/vis/att_imgid_%s_headnum_%s_z_%s.jpg'%(img_idx,head,z),concat_img)
                
            predictions.append(output)
            labels.append(target)
            patient_ids.extend(patient_id)
            pbar.update(args.batch_size)
        pbar.close()

        if args.return_hidden:
            labels_total_ = torch.cat(labels[:-1]).reshape(-1)
            labels_total_ = torch.cat([labels_total_,labels[-1]])
            print("Computing t-SNE embedding")
            X_embedding = np.concatenate(embedding_list_test, axis=0)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)

            X_tsne_2d = tsne2d.fit_transform(X_embedding)
            plot_embedding_2d(X_tsne_2d[:,0:2],labels_total_.cpu().detach().numpy(),"t-SNE 2D",args.results_dir)
    
    false_idx,false_pred_idx = pick_false_sample(predictions,labels)
    pred_false_patients = np.array(patient_ids)[false_idx].tolist()
    false_idx_label = torch.cat(labels).detach().cpu().numpy()[false_idx].tolist()
    false_pred_idx = false_pred_idx.tolist()
    pred_false_patients_dict = {k:(v1,v2) for k,v1,v2 in zip(pred_false_patients,false_idx_label,false_pred_idx)}
    print(pred_false_patients_dict)
    with open(os.path.join(args.results_dir,'pred_false_idxs.txt'),'w') as f_w:
        for k,v in pred_false_patients_dict.items():
            f_w.writelines('%s\t%s\t%s\n'%(k,v[0],v[1]))

    evaluation_metrics,confusion_matrix = compute_metrics(predictions,labels,args)
    print(confusion_matrix)

    output_str = 'Test:\n'
    for key, value in evaluation_metrics.items():
        output_str += f'{key}: {value}\n'
    _logger.info(output_str)
    
    return process_prediction(predictions)

def pick_false_sample(predictions,labels):
    pred = process_prediction(predictions)
    idx = np.argmax(pred,axis=-1)
    labels = torch.cat(labels, dim=0).detach()
    labels = labels.cpu().numpy()
    false_idx = np.where(idx!=labels)[0]

    return false_idx,idx[false_idx]


def process_prediction(outputs):
    outputs = torch.cat(outputs, dim=0).detach()
    pred_score = torch.softmax(outputs, dim=1)
    return pred_score.cpu().numpy()


def write_score2json(score_info, args):
    score_info = score_info.astype(float)
    score_list = []
    anno_info = np.loadtxt(args.val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'prediction': pred,
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    save_name = os.path.join(args.results_dir, args.team_name+'.json')
    file = open(save_name, 'w')
    file.write(json_data)
    file.close()
    _logger.info(f"Prediction has been saved to '{save_name}'.")

def compute_metrics(outputs, targets, args):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()

    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    confusion_matrix_result = confusion_matrix(outputs, targets)
    metrics = OrderedDict([
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
    ])
    
    num_classes = np.unique(targets)
    for i in num_classes:
        index = np.where(targets==i)
        cur_outputs = outputs[index]
        cur_targets = targets[index]
        acc = ACC(cur_outputs, cur_targets)
        print('class:%s,acc=%s'%(i,acc))

    return metrics,confusion_matrix_result

def plot_embedding_2d(X, y, title=None, save_path=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 7.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig(os.path.join(save_path,'tsne.png'))

def main():
    setup_default_logging()
    args = parser.parse_args()
    score = validate(args)
    os.makedirs(args.results_dir, exist_ok=True)
    write_score2json(score, args)


if __name__ == '__main__':
    main()
