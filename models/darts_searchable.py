import torch
import torch.optim as op
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from IPython import embed
import effdet
from data import create_dataset, create_loader, resolve_input_config


# FLIR-Aligned
import models.search.flira_darts_searchable as flira


from models.utils import parse_opts
import models.search.tools as tools


class FlirA_Searcher():
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger


        model_config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        model_config.num_classes = args.num_classes

        input_config = resolve_input_config(args, model_config)

        train_dataset = create_dataset(args.dataset, args.root,'train')
        val_dataset = create_dataset(args.dataset, args.root,'val')
        test_dataset = create_dataset(args.dataset, args.root,'test')

        train_dataloader = create_loader(
            train_dataset,
            input_size=input_config['input_size'],
            batch_size=args.batchsize,
            use_prefetcher=args.prefetcher,
            interpolation=input_config['interpolation'],
            fill_color=input_config['fill_color'],
            rgb_mean=input_config['rgb_mean'],
            rgb_std=input_config['rgb_std'],
            thermal_mean=input_config['thermal_mean'],
            thermal_std=input_config['thermal_std'],
            num_workers=args.workers,
            pin_mem=args.pin_mem,
            is_training=True
        )

        val_dataloader = create_loader(
            val_dataset,
            input_size=input_config['input_size'],
            batch_size=args.batchsize,
            use_prefetcher=args.prefetcher,
            interpolation=input_config['interpolation'],
            fill_color=input_config['fill_color'],
            rgb_mean=input_config['rgb_mean'],
            rgb_std=input_config['rgb_std'],
            thermal_mean=input_config['thermal_mean'],
            thermal_std=input_config['thermal_std'],
            num_workers=args.workers,
            pin_mem=args.pin_mem)

        test_dataloader = create_loader(
            test_dataset,
            input_size=input_config['input_size'],
            batch_size=args.batchsize,
            use_prefetcher=args.prefetcher,
            interpolation=input_config['interpolation'],
            fill_color=input_config['fill_color'],
            rgb_mean=input_config['rgb_mean'],
            rgb_std=input_config['rgb_std'],
            thermal_mean=input_config['thermal_mean'],
            thermal_std=input_config['thermal_std'],
            num_workers=args.workers,
            pin_mem=args.pin_mem)



        self.dataloaders = {
        'train': train_dataloader,
        'dev': val_dataloader,
        'test': test_dataloader
        }


        self.datasets = {
        'train': train_dataset,
        'dev': val_dataset,
        'test': test_dataset
        }

    def search(self):
        best_score, best_genotype = flira.train_darts_model(self.dataloaders,
                                                self.datasets, 
                                                    self.args, 
                                                    self.device, 
                                                    self.logger)
        return best_score, best_genotype
