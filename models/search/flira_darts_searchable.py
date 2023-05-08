import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import torch.optim.lr_scheduler as lr_sc
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

    # bb_path = os.path.join(args.checkpointdir, args.fullbb_path)

    # model.full_backbone.load_state_dict(torch.load(full_bb_path))

    # logger.info("Loading Full Backbone checkpoint: " + full_bb_path)

    # head_path = os.path.join(args.checkpointdir, args.head_path)

    # model.head_net.load_state_dict(torch.load(head_path))

    # logger.info("Loading Head checkpoint: " + head_path)

    checkpoint_path = os.path.join(args.checkpointdir, args.model_path)

    checkpoint = torch.load(checkpoint_path)
    checkpoint_dict = checkpoint["state_dict"]
    net_dict = model.state_dict()
    new_checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in net_dict}
    net_dict.update(new_checkpoint_dict)
    model.load_state_dict(net_dict) 

    logger.info("Loading Model Checkpoint: " + checkpoint_path)

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
    # optimizer = op.Adam(None, lr=args.eta_max, weight_decay=1e-4)
    
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

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = args.num_classes
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel
        self.fusion_levels = args.fusion_levels
        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)
        
        self.thermal_backbone = thermal_det.backbone
        self.rgb_backbone = rgb_det.backbone

        fusion_det = EfficientDet(self.config)
        self.fusion_fpn = fusion_det.fpn
        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        self.fusion_nets = nn.ModuleList()
        self.cin = [48,128,208]
        for i in range(self.fusion_levels):

            self.fusion_nets.append(FusionNetwork(cin = self.cin[i], steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=args))

            self.fusion_nets[i] = self.fusion_nets[i].to(device)



    def forward(self,inputs):

        thermal_x, rgb_x = inputs[0], inputs[1]
        thermal_x, rgb_x = self.thermal_backbone(thermal_x), self.rgb_backbone(rgb_x)


        out = [self.fusion_nets[i]([thermal_x[i], rgb_x[i]]) for i in range(self.fusion_levels)]

        out = self.fusion_fpn(out)

        x_class, x_box = self.fusion_class_net(out), self.fusion_box_net(out)

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

        return central_parameters

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
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel
        self.fusion_levels = args.fusion_levels

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges


        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = args.num_classes

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)
        
        self.thermal_backbone = thermal_det.backbone
        self.rgb_backbone = rgb_det.backbone

        fusion_det = EfficientDet(self.config)
        self.fusion_fpn = fusion_det.fpn
        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        # self._criterion = criterion
        self.cin = [48,128,208]

        self.fusion_nets = nn.ModuleList()

        for i in range(self.fusion_levels):

            self.fusion_nets.append(Found_FusionNetwork(cin = self.cin[i],steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         genotype=self._genotype_list[i]))

            self.fusion_nets[i] = self.fusion_nets[i].to(device)
        


    def forward(self, inputs):

        # self.full_backbone.eval()

        # thermal_list, rgb_list = self.full_backbone(inputs)

        # fusion_level_inputs = {}
        # out = []

        # for i in range(self.fusion_levels):
        #     fusion_level_inputs["Level-"+str(i)] =  []
        #     fusion_level_inputs["Level-"+str(i)] += [item[i] for item in thermal_list[1:]] 
        #     fusion_level_inputs["Level-"+str(i)] += [item[i] for item in rgb_list[1:]]

        #     out.append(self.fusion_nets[i](fusion_level_inputs["Level-"+str(i)]))

        # x_class, x_box = self.head_net(out)

        thermal_x, rgb_x = inputs[0], inputs[1]
        thermal_x, rgb_x = self.thermal_backbone(thermal_x), self.rgb_backbone(rgb_x)

        out = [self.fusion_nets[i]([thermal_x[i], rgb_x[i]]) for i in range(self.fusion_levels)]

        out = self.fusion_fpn(out)

        x_class, x_box = self.fusion_class_net(out), self.fusion_box_net(out)

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


        return central_parameters


    def arch_parameters(self):

        arch_parameters = []

        for i in range(self.fusion_levels):
            arch_parameters.append({'params':self.fusion_nets[i].arch_parameters()})

        return arch_parameters


#############################################################################################################################