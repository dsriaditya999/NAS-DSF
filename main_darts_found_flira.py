import argparse
import time
import re
import glob 
import logging
import sys
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as op
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import models.search.darts.utils as utils
import models.auxiliary.scheduler as sc
import models.search.train_searchable.flira as tr
import models.search.flira_darts_searchable as flira
from models.search.plot_genotype import Plotter
from models.search.darts.genotypes import *
from models.utils import parse_opts

from IPython import embed

import effdet
from effdet.data import resolve_input_config
from data import create_dataset, create_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # experiment directory
    parser.add_argument('root', metavar='DIR',
                        help='path to dataset root')
    parser.add_argument('--search_exp_dir', type=str, help='evaluate which search exp', default=None)
    parser.add_argument('--eval_exp_dir', type=str, help='evaluate which eval exp', default=None)
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

    parser.add_argument('--dataset', default='flir_aligned', type=str, metavar='DATASET',
                        help='Name of dataset (default: "coco"')
    # parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    parser.add_argument('--parallel', help='Use several GPUs', action='store_true', dest='parallel',
                        default=False)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # parser.add_argument('--datadir', type=str, help='data directory',
    #                     default='EgoGesture')
    # parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    # parser.add_argument('--parallel', help='Use several GPUs', action='store_true', dest='parallel',
                        # default=False)
    # parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=32)

    # others
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # basic learning settings
    parser.add_argument('--batchsize', type=int, help='batch size', default=16)
    parser.add_argument('--epochs', type=int, help='training epochs', default=50)
    parser.add_argument("--drpt", action="store", default=0.2, dest="drpt", type=float, help="dropout")

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
    parser.add_argument('--num-classes', type=int, default=90, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    
    # network optimizer and scheduler
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--eta_max', type=float, help='for cosine annealing scheduler, max learning rate', default=0.003)
    parser.add_argument('--eta_min', type=float, help='for cosine annealing scheduler, max learning rate', default=0.000001)
    parser.add_argument('--Ti', type=int, help='for cosine annealing scheduler, epochs Ti', default=5)
    parser.add_argument('--Tm', type=int, help='for cosine annealing scheduler, epochs multiplier Tm', default=2)

    # wandb
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging and visualization')

    parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
    
    return parser.parse_args()

def get_data(args):
    model_config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
    model_config.num_classes = args.num_classes

    input_config = resolve_input_config(args, model_config)

    train_dataset = create_dataset(args.dataset, args.root,'train')
    val_dataset = create_dataset(args.dataset, args.root,'val')
    test_dataset = create_dataset(args.dataset, args.root,'val')

    train_dataloader = create_loader(
        train_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batchsize,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    val_dataloader = create_loader(
        val_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batchsize,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    test_dataloader = create_loader(
        test_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batchsize,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)



    dataloaders = {
    'train': train_dataloader,
    'dev': val_dataloader,
    'test': test_dataloader
    }


    datasets = {
    'train': train_dataset,
    'dev': val_dataset,
    'test': test_dataset
    }
    return dataloaders, datasets

def train_model(model, dataloaders, datasets, args, device, logger):

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize

    if torch.cuda.device_count() > 1 and args.parallel:
        params = model.module.central_params()
    else:
        params = model.central_params()

    # loading pretrained weights

    full_bb_path = os.path.join(args.checkpointdir, args.fullbb_path)

    model.full_backbone.load_state_dict(torch.load(full_bb_path))

    logger.info("Loading Full Backbone checkpoint: " + full_bb_path)

    head_path = os.path.join(args.checkpointdir, args.head_path)

    model.head_net.load_state_dict(torch.load(head_path))

    logger.info("Loading Head checkpoint: " + head_path)

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)

    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    plotter = Plotter(args)

    # status 
    status = 'eval'
    test_metric = tr.train_flira_track_acc(model, None, optimizer, 
                                            scheduler, dataloaders, datasets, dataset_sizes,
                                            device, args.epochs,
                                            args.parallel, logger, plotter, args,
                                            status)

    # logger.info('Final test acc: ' + str(val_acc) )
    return test_metric

def test_model(model, dataloaders, datasets, args, device, 
                logger, test_model_path, genotype):
    # criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(test_model_path))
    print("Weights Loaded!")

    model.eval()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['test']}
    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        # print("using parallel")
        model = torch.nn.DataParallel(model)
    model.to(device)
    # status 
    status = 'eval'
    test_acc = tr.test_ego_track_acc(model, dataloaders, datasets, genotype, 
                                    dataset_sizes, device, logger, args)
    test_acc = test_acc.item()
    # logger.info('Final test accuracy: {}'.format(test_acc))
    return test_acc

if __name__ == "__main__":
    args = parse_args()
    args.prefetcher = not args.no_prefetcher
    # args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    test_only = False

    # test only
    test_model_path = None

    if args.eval_exp_dir != None:
        test_only = True
        eval_exp_dir = args.eval_exp_dir
        args.search_exp_dir = args.eval_exp_dir.split('/')[0]

        batchsize = args.batchsize
        epochs = args.epochs

        search_exp_dir = args.search_exp_dir
        
        args.batchsize = batchsize
        args.epochs = epochs
        
        args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(eval_exp_dir, args.save)
        
        best_test_model_path = os.path.join(eval_exp_dir, 'best', 'best_test_model.pt')
        best_genotype_path = os.path.join(eval_exp_dir, 'best', 'best_test_genotype.pkl')

    elif args.search_exp_dir != None:
        best_genotype_path = os.path.join(args.search_exp_dir, 'best', 'best_genotype.pkl')

        batchsize = args.batchsize
        epochs = args.epochs

        search_exp_dir = args.search_exp_dir
        # new_args = utils.load_pickle(os.path.join(args.search_exp_dir, 'args.pkl'))
        # args = new_args
        
        args.batchsize = batchsize
        args.epochs = epochs
        
        args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(search_exp_dir, args.save)

    args.no_bad_skel = True

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)


    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info("args = %s", args)
    # opt = parse_opts(args)
    # logging.info("opt = %s", opt)
    # embed()

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # criterion = torch.nn.CrossEntropyLoss()
    genotype_list = utils.load_pickle(best_genotype_path)
    # genotype = Genotype(edges=[('skip', 3), ('skip', 7)], steps=[StepGenotype(inner_edges=[('skip', 1), ('skip', 0)], inner_steps=['cat_conv_relu'], inner_concat=[2])], concat=[8])
    
    model = flira.Found_Att_Fusion_Net(args, genotype_list, device)

    dataloaders, datasets = get_data(args)
    start_time = time.time()
     
    model_acc = None
    if test_only:
        model_acc = test_model(model, dataloaders, datasets, args, device, logger, best_test_model_path, genotype_list)
    else:
        model_acc = train_model(model, dataloaders, datasets, args, device, logger)

    time_elapsed = time.time() - start_time
    logger.info("*" * 50)
    logger.info('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Model acc: {}'.format(model_acc))


    # filename = args.checkpointdir+"/final_conf_" + confstr + "_" + str(modelacc.item())+'.checkpoint'
    # torch.save(rmode.state_dict(), filename)
