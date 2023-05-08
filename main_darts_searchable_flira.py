import argparse
import time
import glob 
import logging
import sys
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import models.darts_searchable as S
import models.search.darts.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # experiment directory
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')

    # pretrained backbone checkpoints and annotations
    parser.add_argument('--checkpointdir', type=str, help='pretrained checkpoints and annotations dir',
                        default='checkpoints/flira')
    #parser.add_argument('--annotation', default='egogestureall_but_None.json', type=str, help='Annotation file path')
    parser.add_argument('--fullbb_path', type=str, help='Full Backbone model pth path',
                        default='Att_Fusion_Net_Pretrained_Best.pth.tar')
    parser.add_argument('--head_path', type=str, help='Head model pth path',
                        default='Head_Net_Pretrained_Best.pth.tar')
    # parser.add_argument('--depth_cp', type=str, help='depth video model pth path',
    #                     default='egogesture_resnext_1.0x_Depth_32_acc_93.61060.pth')
    
    # dataset and data parallel
    parser.add_argument('root', metavar='DIR',
                        help='path to dataset root')
    parser.add_argument('--dataset', default='flir_aligned', type=str, metavar='DATASET',
                        help='Name of dataset (default: "coco"')
    # parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    parser.add_argument('--parallel', help='Use several GPUs', action='store_true', dest='parallel',
                        default=False)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # others
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # basic learning settings
    parser.add_argument('--batchsize', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--epochs', type=int, help='training epochs', default=30)
    parser.add_argument("--drpt", action="store", default=0, dest="drpt", type=float, help="dropout")

    # number of input features
    parser.add_argument('--num_input_nodes', type=int, help='total number of modality features', default=10)
    parser.add_argument('--num_keep_edges', type=int, help='cells and steps will have 2 input edges', default=2)
    
    # for cells and steps and inner representation size
    parser.add_argument('--C', type=int, help='channels', default=128)
    parser.add_argument('--L', type=int, help='length after pool', default=8)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=2)
    parser.add_argument('--steps', type=int, help='cell steps', default=2)
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=3)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=2)
    parser.add_argument('--fusion_levels', type=int, help='Fusion Levels', default=5)

    # number of classes    
    # parser.add_argument('--num_outputs', type=int, help='output dimension', default=83)
    parser.add_argument('--num-classes', type=int, default=90, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    
    # network optimizer and scheduler
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--eta_max', type=float, help='for cosine annealing scheduler, max learning rate', default=0.003)
    parser.add_argument('--eta_min', type=float, help='for cosine annealing scheduler, max learning rate', default=0.000001)
    parser.add_argument('--Ti', type=int, help='for cosine annealing scheduler, epochs Ti', default=5)
    parser.add_argument('--Tm', type=int, help='for cosine annealing scheduler, epochs multiplier Tm', default=2)

    # wandb
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging and visualization')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.prefetcher = not args.no_prefetcher

    # for reproductivity
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    cudnn.deterministic = True
    cudnn.benchmark = False

    torch.cuda.manual_seed(args.seed)

    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('EXP_CBAM/flira', args.save)
    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info("args = %s", args)

    # hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # searcher
    flira_searcher = S.FlirA_Searcher(args, device, logger)

    # search
    logger.info("NAS-DSF for FLIR-Aligned Started.")
    start_time = time.time()
    best_score, best_genotype = flira_searcher.search()
    time_elapsed = time.time() - start_time

    logger.info("*" * 50)
    logger.info('Search complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Now listing best fusion_net genotype:')

    for i in range(len(best_genotype)):
        logger.info(best_genotype[i])

