import torch
torch.autograd.set_detect_anomaly(True)
import models.auxiliary.scheduler as sc
import copy
from tqdm import tqdm
import os
import time
from models.search.darts.utils import count_parameters, save, save_pickle
import wandb
from .detector import DetBenchTrainImagePair, DetBenchPredictImagePair
from .utils.evaluator import CocoEvaluator
from .utils.utils import visualize_detections, visualize_target
from effdet import create_evaluator
from IPython import embed
from collections import Counter
from timm.utils import AverageMeter
import pickle



def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer in name:
            param.requires_grad = False

def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer in name:
            module.eval()

# train model with darts mixed operations (in Search Phase)
def train_flira_track_acc(model, architect, optimizer, scheduler, dataloaders, datasets, dataset_sizes,
                device=None, num_epochs=200, parallel=False, logger=None,
                plotter=None, args=None, status='search'):


    # Initializations

    best_genotype = None
    best_metric = 0
    best_epoch = 0
    
    best_test_genotype = None
    best_test_metric = 0
    best_test_epoch = 0
    # best_test_model_sd = copy.deepcopy(model.state_dict())


    # wandb setup - logging 

    if args.wandb:
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='NAS-DSF',
          config=config
        )

    # Freeze the Backbone

    freeze(model,"full_backbone")

    # Freeze the Head Networks

    freeze(model,"head_net")

    # Create a Training Bench

    training_bench = DetBenchTrainImagePair(model, create_labeler=True)

    if torch.cuda.device_count() > 1 and args.parallel:
        training_bench = torch.nn.DataParallel(training_bench)
        # model = torch.nn.parallel.DistributedDataParallel(model)
    training_bench.to(device)

    # COCO Evaluator

    if status == "search":

        evaluator = CocoEvaluator(datasets['dev'],logger, distributed=False, pred_yxyx=False)

    else:

        evaluator = CocoEvaluator(datasets['test'],logger, distributed=False, pred_yxyx=False)


    for epoch in range(1, num_epochs+1):
        # Each epoch has a training and validation phase
        logger.info("Epoch: {}".format(epoch) )
        logger.info("EXP: {}".format(args.save) )

        phases = []
        if status == 'search':
            phases = ['train', 'dev']
        else:
            # here the train is train + dev
            phases = ['train', 'test']

        for phase in phases:
            if phase == 'train':
                if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                    scheduler.step()
                training_bench.train()  # Set model to training mode
                set_eval_mode(training_bench, "full_backbone") 
            elif phase == 'dev':
                if architect is not None:
                    architect.log_learning_rate(logger)
                training_bench.train()  # Set model to training mode
                set_eval_mode(training_bench, "full_backbone") 
            else:
                training_bench.eval()  # Set model to evaluate mode

            running_loss = 0.0

            for param_group in optimizer.param_groups:
                logger.info("Learning Rate: {}".format(param_group['lr']))
                break

            with tqdm(dataloaders[phase]) as t:
                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    thermal_img_tensor, rgb_img_tensor, target = data[0], data[1], data[2]
                    
                    # device
                    thermal_img_tensor = thermal_img_tensor.to(device)
                    rgb_img_tensor = rgb_img_tensor.to(device)
                    # target = target.to(device)

                    # updates darts cell
                    if status == 'search' and (phase == 'dev' or phase == 'test'):
                        if architect is not None:
                            architect.step(thermal_img_tensor, rgb_img_tensor, target)


                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    flag = (phase == 'train' or (phase == 'dev' and status == 'eval'))
                    with torch.set_grad_enabled(flag):
                        output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass= not flag)
                        loss = output['loss']
                        # backward + optimize only if in training phase
                        if flag:
                            if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                scheduler.step()
                                scheduler.update_optimizer(optimizer)
                            loss.backward()
                            optimizer.step()

                            if args.wandb:
                                visualize_target(datasets['train'], target, wandb, args, 'train_'+status)

                    # statistics
                    running_loss += loss.item() * rgb_img_tensor.size(0)

                    if (phase=='dev' and status=='search') or (phase=='test' and status=='eval'):

                        evaluator.add_predictions(output['detections'], target)
                        
                        if args.wandb and (epoch == args.epochs) and (status=='search'):
                            visualize_detections(datasets['dev'], output['detections'], target, wandb, args, 'val_'+status)

                        if args.wandb and (epoch == args.epochs) and (status=='eval'):
                            visualize_detections(datasets['test'], output['detections'], target, wandb, args, 'test_'+status)

                    postfix_str = 'batch_loss: {:.03f}'.format(loss.item())

                    t.set_postfix_str(postfix_str)
                    t.update()

            epoch_loss = running_loss / dataset_sizes[phase]

            if (phase=='dev' and status=='search') or (phase=='test' and status=='eval'):

                epoch_metric  = evaluator.evaluate()

                logger.info('{} Loss: {:.4f} mAP @[IoU=0.50:0.95] : {:.4f}'.format(
                    phase, epoch_loss, epoch_metric))

            if phase=='train':

                logger.info('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
            
            print('dataset_size:', dataset_sizes[phase])

            genotype = None

            if parallel:
                num_params = 0
                for i in range(training_bench.module.model.fusion_levels):
                    num_params += count_parameters(training_bench.module.model.fusion_nets[i])

                logger.info("Fusion Nets Params: {}".format(num_params) )

                genotype = training_bench.module.model.genotype()
            else:
                fusion_net_params = 0
                for i in range(training_bench.model.fusion_levels):
                    fusion_net_params += count_parameters(training_bench.model.fusion_nets[i])

                full_backbone_params = count_parameters(training_bench.model.full_backbone)
                head_net_params = count_parameters(training_bench.model.head_net)
                full_params = count_parameters(training_bench.model)

                # total_params = sum(p.numel() for p in training_bench.model.parameters())
                # total_trainable_params = sum(p.numel() for p in training_bench.model.parameters() if p.requires_grad)

                print("*"*50)
                logger.info("Full Backbone Params : {}".format(full_backbone_params) )
                logger.info("Head Network Params : {}".format(head_net_params) )
                logger.info("Fusion Nets Params : {}".format(fusion_net_params) )
                logger.info("Total Model Parameters : {}".format(full_params) )
                print("*"*50)

                genotype = training_bench.model.genotype()


            for i in range(len(genotype)):
                logger.info(str(genotype[i]))
            
            # deep copy the model
            if phase == 'dev' and epoch_metric >= best_metric:
                best_metric = epoch_metric
                # best_test_model_sd = copy.deepcopy(model.state_dict())
                best_genotype = copy.deepcopy(genotype)
                best_epoch = epoch

                if parallel:
                    save(model, os.path.join(args.save, 'best', 'best_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                for i in range(len(best_genotype)):
                    file_name = "best_epoch_"+str(epoch)+"_level_"+str(i)
                    file_name = os.path.join(args.save, "architectures", file_name)
                    plotter.plot(genotype[i], file_name, task='flira')

                best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                save_pickle(best_genotype, best_genotype_path)
                        
            # deep copy the model
            if phase == 'test' and epoch_metric >= best_test_metric:
                best_test_metric = epoch_metric
                # best_test_model_sd = copy.deepcopy(model.state_dict())
                best_test_genotype = copy.deepcopy(genotype)
                best_test_epoch = epoch
                best_test_model = copy.deepcopy(model)

                if parallel:
                    save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                save_pickle(best_test_genotype, best_test_genotype_path)

        

    if status=="search":

        logger.info("Current best dev accuracy: {}, at training epoch: {}".format(best_metric, best_epoch) )
        return best_metric, best_genotype

    else:

        best_test_model.eval()
        bench = DetBenchPredictImagePair(best_test_model)
        # bench = DetBenchTrainImagePair(model, create_labeler=True)

        bench.to(device)
        bench.eval()

        evaluator = create_evaluator(args.dataset, datasets['test'], pred_yxyx=False)

        phase = 'test'

        with torch.no_grad():
            with tqdm(dataloaders[phase]) as t:
                # Iterate over data.
                for data in dataloaders[phase]:

                    # get the inputs
                    thermal_img_tensor, rgb_img_tensor, target = data[0], data[1], data[2]
                    
                    # device
                    thermal_img_tensor = thermal_img_tensor.to(device)
                    rgb_img_tensor = rgb_img_tensor.to(device)
                    # target = target.to(device)

                    # output = bench(thermal_img_tensor, rgb_img_tensor, target,eval_pass=True)
                    output = bench(thermal_img_tensor, rgb_img_tensor, img_info=target)
                    evaluator.add_predictions(output, target)

                    t.update()

        mean_ap = 0.
        if datasets['test'].parser.has_labels:
            mean_ap = evaluator.evaluate(output_result_file=args.results)
            # mean_ap = CocoEvaluator(dataset, distributed=distributed, pred_yxyx=pred_yxyx)
        else:
            evaluator.save(args.results)

    # mean_ap  = evaluator.evaluate()




        print("*"*50)
        print("Mean Average Precision Obtained is : "+str(mean_ap))
        print("*"*50)
        
        # for i in range(len(best_test_genotype)):
        #     logger.info(str(best_test_genotype[i]))

   
        logger.info("Current best test accuracy: {}, at training epoch: {}".format(best_test_metric, best_test_epoch) )
        return best_test_metric
        

def test_ego_track_acc(model, dataloaders, datasets, genotype_list, 
                        dataset_sizes, device, logger, args):
    model.eval()
    bench = DetBenchPredictImagePair(model)
    # bench = DetBenchTrainImagePair(model, create_labeler=True)

    bench.to(device)
    bench.eval()

    evaluator = create_evaluator(args.dataset, datasets['test'], pred_yxyx=False)
    # evaluator = CocoEvaluator(datasets['test'], distributed=False, pred_yxyx=False)

    logger.info("EXP: {}".format(args.save) )
    phase = 'test'

    with torch.no_grad():
        with tqdm(dataloaders[phase]) as t:
            # Iterate over data.
            for data in dataloaders[phase]:

                # get the inputs
                thermal_img_tensor, rgb_img_tensor, target = data[0], data[1], data[2]
                
                # device
                thermal_img_tensor = thermal_img_tensor.to(device)
                rgb_img_tensor = rgb_img_tensor.to(device)
                # target = target.to(device)

                # output = bench(thermal_img_tensor, rgb_img_tensor, target,eval_pass=True)
                output = bench(thermal_img_tensor, rgb_img_tensor, img_info=target)
                evaluator.add_predictions(output, target)

                t.update()

    mean_ap = 0.
    if datasets['test'].parser.has_labels:
        mean_ap = evaluator.evaluate(output_result_file=args.results)
        # mean_ap = CocoEvaluator(dataset, distributed=distributed, pred_yxyx=pred_yxyx)
    else:
        evaluator.save(args.results)

    # mean_ap  = evaluator.evaluate()




    print("*"*50)
    print("Mean Average Precision Obtained is : "+str(mean_ap))
    print("*"*50)
    
    for i in range(len(genotype_list)):
        logger.info(str(genotype_list[i]))


    return mean_ap
