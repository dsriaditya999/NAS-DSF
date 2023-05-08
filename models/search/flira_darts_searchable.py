import torch
import torch.nn as nn
import torch.optim as op
import os


import torch.optim.lr_scheduler as lr_sc
import models.auxiliary.aux_models as aux
import models.central.ego as ego
import models.search.train_searchable.flira as tr
import torch.nn.functional as F

from IPython import embed
import numpy as np

from .darts.model_search import FusionNetwork
from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from .darts.architect import Architect


import logging
import math
from collections import OrderedDict
from functools import partial
from typing import List, Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import create_conv2d, create_pool2d, get_act_layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import load_checkpoint
import effdet
from effdet import EfficientDet
from effdet.efficientdet import get_feature_info
from effdet import DetBenchTrain
import math
from copy import deepcopy

import itertools

from omegaconf import OmegaConf


# from .config import get_fpn_config

_DEBUG = False
_USE_SCALE = False
_ACT_LAYER = get_act_layer('silu')




def train_darts_model(dataloaders, datasets, args, device, logger):

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize

    # model to train
    model = Searchable_Att_Fusion_Net(args,device)

    params = model.central_params()

    # loading pretrained weights

    full_bb_path = os.path.join(args.checkpointdir, args.fullbb_path)

    model.full_backbone.load_state_dict(torch.load(full_bb_path))

    logger.info("Loading Full Backbone checkpoint: " + full_bb_path)

    head_path = os.path.join(args.checkpointdir, args.head_path)

    model.head_net.load_state_dict(torch.load(head_path))

    logger.info("Loading Head checkpoint: " + head_path)

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max/0.95, weight_decay=1e-4)
    # optimizer = op.Adam(None, lr=args.eta_max, weight_decay=1e-4)
    # scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
    #                                           num_batches_per_epoch)

    scheduler = lr_sc.ExponentialLR(optimizer, gamma=0.95)

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)
    architect = Architect(model, args, arch_optimizer, device)

    plotter = Plotter(args)

    best_score, best_genotype = tr.train_flira_track_acc(model, architect,
                                            optimizer, scheduler, dataloaders, datasets,
                                            dataset_sizes,
                                            device=device, 
                                            num_epochs=args.epochs, 
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args)

    return best_score, best_genotype

#############################################################################################################################
########################################## Searchable Attention Fusion Network ##############################################

class Searchable_Att_Fusion_Net(nn.Module):
    def __init__(self,args,device):

        super().__init__()

        self.full_backbone = Att_Fusion_Net(args.num_classes)

        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel
        self.fusion_levels = args.fusion_levels

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        # self._criterion = criterion

        self.fusion_nets = nn.ModuleList()

        for i in range(self.fusion_levels):

            self.fusion_nets.append(FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=args))

            self.fusion_nets[i] = self.fusion_nets[i].to(device)

        self.head_net = NAS_Head_Net(args.num_classes)
        self.head_net =self.head_net.to(device)


    def forward(self,inputs):

        self.full_backbone.eval()

        thermal_list, rgb_list = self.full_backbone(inputs)

        fusion_level_inputs = {}
        out = []

        for i in range(self.fusion_levels):
            fusion_level_inputs["Level-"+str(i)] =  []
            fusion_level_inputs["Level-"+str(i)] += [item[i] for item in thermal_list[1:]] 
            fusion_level_inputs["Level-"+str(i)] += [item[i] for item in rgb_list[1:]]

            out.append(self.fusion_nets[i](fusion_level_inputs["Level-"+str(i)]))

        x_class, x_box = self.head_net(out)

        return x_class, x_box


    def genotype(self):

        genotypes = []

        for i in range(self.fusion_levels):
            genotypes.append(self.fusion_nets[i].genotype())

        return genotypes

    
    def central_params(self):
        central_parameters = []

        for i in range(self.fusion_levels):
            central_parameters.append({'params':self.fusion_nets[i].parameters()})

        # central_parameters.append({'params':self.head_net.parameters()})

        return central_parameters

    
    def _loss(self, input_features, labels):
        # logits = self(input_features)
        # return self._criterion(logits, labels) 
        pass

    def arch_parameters(self):

        arch_parameters = []

        for i in range(self.fusion_levels):
            arch_parameters.append({'params':self.fusion_nets[i].arch_parameters()})

        return arch_parameters

#############################################################################################################################
########################################## Found Attention Fusion Network ##############################################

class Found_Att_Fusion_Net(nn.Module):
    def __init__(self, args, genotype_list, device):
        super().__init__()

        self.args = args


        self._genotype_list = genotype_list

        self.full_backbone = Att_Fusion_Net(args.num_classes)

        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel
        self.fusion_levels = args.fusion_levels

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        # self._criterion = criterion

        self.fusion_nets = nn.ModuleList()

        for i in range(self.fusion_levels):

            self.fusion_nets.append(Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         genotype=self._genotype_list[i]))

            self.fusion_nets[i] = self.fusion_nets[i].to(device)

        self.head_net = NAS_Head_Net(args.num_classes)
        self.head_net =self.head_net.to(device)
        


    def forward(self, inputs):

        self.full_backbone.eval()

        thermal_list, rgb_list = self.full_backbone(inputs)

        fusion_level_inputs = {}
        out = []

        for i in range(self.fusion_levels):
            fusion_level_inputs["Level-"+str(i)] =  []
            fusion_level_inputs["Level-"+str(i)] += [item[i] for item in thermal_list[1:]] 
            fusion_level_inputs["Level-"+str(i)] += [item[i] for item in rgb_list[1:]]

            out.append(self.fusion_nets[i](fusion_level_inputs["Level-"+str(i)]))

        x_class, x_box = self.head_net(out)

        return x_class, x_box

    
    def genotype(self):

        genotypes = []

        for i in range(self.fusion_levels):
            genotypes.append(self.fusion_nets[i]._genotype)

        return genotypes

    
    def central_params(self):
        central_parameters = []

        for i in range(self.fusion_levels):
            central_parameters.append({'params':self.fusion_nets[i].parameters()})

        # central_parameters.append({'params':self.head_net.parameters()})

        return central_parameters

    
    def _loss(self, input_features, labels):
        # logits = self(input_features)
        # return self._criterion(logits, labels) 
        pass

    def arch_parameters(self):

        arch_parameters = []

        for i in range(self.fusion_levels):
            arch_parameters.append({'params':self.fusion_nets[i].arch_parameters()})

        return arch_parameters

#############################################################################################################################
########################################## Helper Functions for Attention Fusion Network ####################################

class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.
    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)
    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False)


class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self, in_channels, out_channels, input_size, output_size, pad_type='',
            downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_bn=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module('conv', ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias, act_layer=None))

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type)
            else:
                if _USE_SCALE:  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module('downsample', down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
                else:
                    self.add_module('upsample', Interpolate2d(size=output_size, mode=upsample))


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_channels, inputs_offsets, output_size, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]['num_chs'], fpn_channels,
                input_size=feature_info[offset]['size'], output_size=output_size, pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, feat_sizes, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER,
                 apply_resample_bn=False, pre_act=True, separable_conv=True, redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc['feat_level']]) for fc in fpn_config.nodes]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                output_size=feat_sizes[fnode_cfg['feat_level']], pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=False, norm_layer=norm_layer, act_layer=act_layer)
            if pre_act:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
            
#         for temp in x[-self.num_levels::]:
#             print(temp.size())
#         print("")
            
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size
            

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x



class NAS_BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(NAS_BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size
            


        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        

        
        for resample in self.resample.values():
            x.append(resample(x[-1]))

        x_list = []

        x_list.append(x)
        
        for layer in self.cell:
            
            x = layer([torch.clone(item) for item in x])
            x_list.append(x)
            
        return x,x_list

def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))
    return p


def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.
    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.
    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = [level_last_id(i), level_last_id(i + 1)] if i != max_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = [level_last_id(i), level_last_id(i - 1)] if i != min_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    return p


def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.
    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).
    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))

    i = max_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [
                level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][2], node_ids[i][4]],
            'weight_method': quad_method
        })
        node_ids[i].append(next(id_cnt))

    return p


def get_fpn_config(fpn_name, min_level=3, max_level=7):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'bifpn_attn': bifpn_config(min_level=min_level, max_level=max_level, weight_method='attn'),
        'bifpn_fa': bifpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'pan_sum': panfpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'pan_fa': panfpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'qufpn_sum': qufpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'qufpn_fa': qufpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    }
    return name_to_config[fpn_name]


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes

def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(backbone.feature_info())]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
    return feature_info

#################################################################################################################################
################################################# Attention Fusion Network ######################################################

class Att_Fusion_Net(nn.Module):

    def __init__(self,num_classes):
        super(Att_Fusion_Net, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = num_classes

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)

        self.thermal_backbone = thermal_det.backbone
        thermal_feature_info = get_feature_info(self.thermal_backbone)
        self.thermal_fpn = NAS_BiFpn(self.config,thermal_feature_info)


        self.rgb_backbone = rgb_det.backbone
        rgb_feature_info = get_feature_info(self.rgb_backbone)
        self.rgb_fpn = NAS_BiFpn(self.config,rgb_feature_info)


    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]
        
        thermal_x, rgb_x = self.thermal_backbone(thermal_x), self.rgb_backbone(rgb_x)
        
        _, thermal_list = self.thermal_fpn(thermal_x)
        
        _, rgb_list = self.rgb_fpn(rgb_x)

        return thermal_list, rgb_list

#################################################################################################################################
#################################################################################################################################

class NAS_Head_Net(nn.Module):

    def __init__(self,num_classes):
        super(NAS_Head_Net, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = num_classes

        fusion_det = EfficientDet(self.config)
        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

    def forward(self, x):
        
        return self.fusion_class_net(x), self.fusion_box_net(x)


#################################################################################################################################

class Searchable_RGB_Depth_Net(nn.Module):
    def __init__(self, args, opt, criterion):
        super().__init__()

        self.args = args
        self.opt = opt
        self.criterion = criterion

        self.rgb_net = ego.get_rgb_model(opt)
        self.depth_net = ego.get_depth_model(opt)

        self.reshape_layers = self.create_reshape_layers(args)
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        
        self._criterion = criterion

        self.fusion_net = FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)
        # self.bn = nn.BatchNorm1d(args.num_outputs * 2)
        # self.central_classifier = nn.Linear(args.num_outputs * 2,
        #                                     args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 512, 1024, 2048, 2048]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, inputs):
        # rgb = inputs[:,0:3,:,:,:]
        # depth = inputs[:,3::,:,:,:]
        rgb, depth = inputs
        # apply net on input rgb videos 
        self.rgb_net.eval()       
        rgb_features = self.rgb_net(rgb)
        rgb_features = rgb_features[0:-1]

        # # apply net on input depth videos   
        self.depth_net.eval()     
        depth_features = self.depth_net(depth)
        depth_features = depth_features[0:-1]
        # embed()
        # exit(0)
        input_features = list(rgb_features) + list(depth_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        # exit(0)
        
        # out = rgb_features[-1]
        # out = depth_features[-1]
        # out = depth_features[-1]
        # out = torch.cat([rgb_features[-1], depth_features[-1]], dim=1)

        # out = torch.cat([input_features[3], input_features[7]], dim=1)

        # out = self.bn(out)
        # out = F.relu(out)
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # out = self.central_classifier(out)
        # , rgb_features[-1]), dim=1)
        # out = F.softmax(out, dim=1)
        # out = self.bn(out)
        # out = F.relu(out)
        # out = self.central_classifier(out)

        # embed()
        # out = F.softmax(out, dim=1)
        # print(out.shape)
        return out

    def genotype(self):
        return self.fusion_net.genotype()
    
    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
            ,{'params': self.reshape_layers.parameters()}
        ]
        return central_parameters
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 


class Found_RGB_Depth_Net(nn.Module):
    def __init__(self, args, opt, criterion, genotype):
        super().__init__()

        self.args = args
        self.opt = opt
        self.criterion = criterion

        self.rgb_net = ego.get_rgb_model(opt)
        self.depth_net = ego.get_depth_model(opt)

        self._genotype = genotype

        for p in self.rgb_net.parameters():
            p.requires_grad = False
        
        for p in self.depth_net.parameters():
            p.requires_grad = False

        self.reshape_layers = self.create_reshape_layers(args)
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        
        self._criterion = criterion

        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         genotype=self._genotype)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)
        # self.bn = nn.BatchNorm1d(args.num_outputs * 2)
        # self.central_classifier = nn.Linear(args.num_outputs * 2,
        #                                     args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 512, 1024, 2048, 2048]
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes))

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
            else:
                reshape_layers.append(nn.ReLU())
        
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, inputs):
        # rgb = inputs[:,0:3,:,:,:]
        # depth = inputs[:,3::,:,:,:]
        rgb, depth = inputs
        # apply net on input rgb videos 
        self.rgb_net.eval()       
        rgb_features = self.rgb_net(rgb)
        rgb_features = rgb_features[0:-1]

        # # apply net on input depth videos   
        self.depth_net.eval()     
        depth_features = self.depth_net(depth)
        depth_features = depth_features[0:-1]
        # embed()
        # exit(0)
        input_features = list(rgb_features) + list(depth_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        # exit(0)
        return out
    
    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
            ,{'params': self.reshape_layers.parameters()}
        ]
        return central_parameters
    
    def genotype(self):
        return self._genotype
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 
