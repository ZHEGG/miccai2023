# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
import numpy as np

layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_visualization = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_visualization:
            attn_copy = attn.clone().detach().cpu().numpy()
            attn_copy = np.sum(attn_copy,axis=2) # N,head_num,Z*H*W


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_visualization:
            return x, attn_copy
        return x, None


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x, return_visualization = False):
        x = x + self.pos_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
            x_attn = self.drop_path(self.gamma_1 * x_attn)
            x = x_attn + x
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
            x_attn = self.drop_path(x_attn)
            x = x_attn + x
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W )
        return x, visualization_heads
   

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
    

class UniFormer_ori(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False, pretrained_cfg_overlay=None, return_visualization = False,
                 return_hidden = False, modified = False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            # self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            # self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2])
            # self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3])
    
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2))

        else:
            if modified:
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size, patch_size=(3, 2, 2), in_chans=in_chans, embed_dim=embed_dim[0])
                self.patch_embed1.proj = nn.Conv3d(in_chans,embed_dim[0],(3,2,2),(3,2,2),groups=in_chans)
            else:
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size, patch_size=(1, 2, 2), in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=(1, 2, 2), in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=(1, 2, 2), in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=(1, 2, 2), in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(1, 2, 2))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        if modified:
            self.head = nn.Sequential(
                nn.Linear(embed_dim[-1],512),
                nn.ReLU(),
                nn.Linear(512,num_classes)
            ) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.return_visualization = return_visualization
        self.return_hidden = return_hidden
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # if isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        N,T,Z,H,W = x.shape
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x,_ = blk(x)
        x = self.patch_embed4(x)
        for i, blk in enumerate(self.blocks4):
            if i < len(self.blocks4)-1:
                x, visualization_heads = blk(x)
            else:
                x, visualization_heads = blk(x, return_visualization = self.return_visualization)
        
        if self.return_visualization:
            depth, height, width = Z,int(H/16),int(W/16)
            visualization_heads = visualization_heads.reshape((visualization_heads.shape[0],visualization_heads.shape[1],depth, height, width))
        x = self.norm(x)
        x = self.pre_logits(x)
        return x,visualization_heads

    def forward(self, x):
        visualizations = []
        x, visualization_heads = self.forward_features(x)
        if self.return_visualization:
            visualizations.append(visualization_heads)
        x = x.flatten(2).mean(-1)
        before_head = x
        x = self.head(x)
        if self.return_hidden:
            return x, visualization_heads, before_head
        return x, visualizations
    
@register_model
def uniformer_small_original(num_classes=2, 
                       num_phase=8,
                       img_size = 224,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = UniFormer_ori(
        in_chans=num_phase,
        num_classes=num_classes,
        img_size=img_size,
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwards)
    model.default_cfg = _cfg()
    if pretrained:
        # default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_small_k400_8x8.pth')
        default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_small_k400_16x8.pth')
        # default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_small_k600_16x8.pth')
        model_para = torch.load(default_pretrained_path)

        for para_name in list(model_para.keys()):
            if 'patch_embed1' in para_name:
                model_para.pop(para_name)
            if 'head' in para_name:
                model_para.pop(para_name)

        keep_state = {k:v for k,v in model_para.items() if k in model.state_dict().keys()}

        missing_state = {k:v for k,v in model.state_dict().items() if k not in model_para.keys()}

        cur_model_state = model.state_dict()
        cur_model_state.update(keep_state)

        model.load_state_dict(cur_model_state)

        # model.load_state_dict(model_para,strict=False)

    return model

@register_model
def uniformer_base_original(num_classes=2, 
                       num_phase=8,
                       img_size = 224,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = UniFormer_ori(
        in_chans=num_phase,
        num_classes=num_classes,
        img_size = img_size,
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwards)
    model.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=(2,2,2), in_chans=num_phase, embed_dim=model.embed_dim[0])
    
    model.default_cfg = _cfg()
    if pretrained:
        default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_base_k600_16x8.pth')
        # default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_base_k400_32x4.pth')
        # default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_base_k400_16x4.pth')
        model_para = torch.load(default_pretrained_path)

        for para_name in list(model_para.keys()):
            if 'patch_embed1' in para_name:
                model_para.pop(para_name)
            if 'head' in para_name:
                model_para.pop(para_name)

        keep_state = {k:v for k,v in model_para.items() if k in model.state_dict().keys()}

        missing_state = {k:v for k,v in model.state_dict().items() if k not in model_para.keys()}

        cur_model_state = model.state_dict()
        cur_model_state.update(keep_state)

        model.load_state_dict(cur_model_state)

        # model.load_state_dict(model_para,strict=False)

    return model

@register_model
def uniformer_xs_original(num_classes=2, 
                       num_phase=8,
                       img_size = 224,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = UniFormer_ori(
        in_chans=num_phase,
        num_classes=num_classes,
        img_size = img_size,
        depth=[3, 5, 9, 3],
        embed_dim=[64, 128, 256, 512], head_dim=32, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwards)
    
    model.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=(4,4,4), in_chans=num_phase, embed_dim=model.embed_dim[0])
    model.default_cfg = _cfg()
    if pretrained:
        default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_xs32_192_k400.pth')
        model_para = torch.load(default_pretrained_path)

        for para_name in list(model_para.keys()):
            if 'patch_embed1' in para_name:
                model_para.pop(para_name)
            if 'head' in para_name:
                model_para.pop(para_name)

        keep_state = {k:v for k,v in model_para.items() if k in model.state_dict().keys()}

        missing_state = {k:v for k,v in model.state_dict().items() if k not in model_para.keys()}

        cur_model_state = model.state_dict()
        cur_model_state.update(keep_state)

        model.load_state_dict(cur_model_state)

        # model.load_state_dict(model_para,strict=False)

    return model

@register_model
def uniformer_xxs_original(num_classes=2, 
                       num_phase=8,
                       img_size = 224,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = UniFormer_ori(
        in_chans=num_phase,
        num_classes=num_classes,
        img_size = img_size,
        depth=[2, 5, 8, 2],
        embed_dim=[56, 112, 224, 448], head_dim=28, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwards)
    
    model.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=(1,2,2), in_chans=num_phase, embed_dim=model.embed_dim[0])
    model.default_cfg = _cfg()
    if pretrained:
        default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_xxs16_128_k400.pth')
        # default_pretrained_path = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/pretrained_model/uniformer_xxs32_160_k400.pth')
        model_para = torch.load(default_pretrained_path)

        for para_name in list(model_para.keys()):
            if 'patch_embed1' in para_name:
                model_para.pop(para_name)
            if 'head' in para_name:
                model_para.pop(para_name)

        keep_state = {k:v for k,v in model_para.items() if k in model.state_dict().keys()}

        missing_state = {k:v for k,v in model.state_dict().items() if k not in model_para.keys()}

        cur_model_state = model.state_dict()
        cur_model_state.update(keep_state)

        model.load_state_dict(cur_model_state)

        # model.load_state_dict(model_para,strict=False)

    return model