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

import models
from metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset

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
parser.add_argument('--model', '-m', metavar='NAME', type=str, nargs='+', default='resnet50',
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
parser.add_argument('--checkpoint', default='', type=str, nargs='+', metavar='PATH',
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
parser.add_argument('--return_glb', action='store_true', default=False, help='if return_glb_input')
parser.add_argument('--modified', action='store_true', default=False, help='if use modified uniformer')

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

    if isinstance(args.model,list):
        assert isinstance(args.checkpoint,list),'if multimodel, must have corresponding checkpoints'
        assert len(args.model) == len(args.checkpoint)
        model_list = []
        for model_name,checkpoint_path in zip(args.model,args.checkpoint):
            # create model
            model = create_model(
                model_name,
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                img_size = args.crop_size[-1],
                pretrained_cfg=None,
                modified = args.modified,
                )
            if args.num_classes is None:
                assert hasattr(
                    model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
                args.num_classes = model.num_classes
            
            load_checkpoint(model, checkpoint_path, args.use_ema)

            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (model_name, param_count))
            
            model = model.cuda()

            if args.apex_amp:
                model = amp.initialize(model, opt_level='O1')

            if args.num_gpu > 1:
                model = torch.nn.DataParallel(
                    model, device_ids=list(range(args.num_gpu)))
            
            model.eval()

            model_list.append(model)
    else:
        # create model
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            img_size = args.crop_size[-1],
            pretrained_cfg=None,
            modified = args.modified,
                            )
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
            
        model.eval()

    dataset = MultiPhaseLiverDataset(args, is_training=False)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=args.pin_mem,
                        shuffle=False)

    predictions_total = []
    labels_total = []

    for idx,model in enumerate(model_list):
        model_name = args.model[idx]
        predictions,labels = val_oridataset(model,model_name,dataset,loader,args)
        predictions_total.append(predictions)
        labels_total.append(labels)

    predictions_total_ = [torch.cat(i) for i in predictions_total]
    predictions_total_ = torch.stack(predictions_total_).mean(dim=0)

    return process_prediction(predictions_total_)

def val_oridataset(model, model_name, dataset, loader, args):
    amp_autocast = suppress

    predictions = []
    labels = []

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
                if model_name == "uniformer_small_original" or model_name == "uniformer_base_original" or model_name == "uniformer_xs_original" or model_name == "uniformer_xxs_original" :
                    output,visualizations = model(input)
                else:
                    raise ValueError('invalid model input')

            predictions.append(output)
            labels.append(target)
            pbar.update(args.batch_size)
        pbar.close()

    return predictions, labels

def process_prediction(outputs):
    if isinstance(outputs,list):
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


def main():
    setup_default_logging()
    args = parser.parse_args()
    score = validate(args)
    os.makedirs(args.results_dir, exist_ok=True)
    write_score2json(score, args)


if __name__ == '__main__':
    main()