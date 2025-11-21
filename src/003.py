# %%
import torch
import torch.optim as optim
import argparse
from torch import nn
import sys, os

# Add the 'modules/deepspeech/src/' directory to the system path
sys.path.insert(0, os.path.abspath('../modules/deepspeech/src'))

# from deepspeech.networks.utils import OverLastDim
# from deepspeech.data import preprocess
from torchvision.transforms import Compose
# import torch.utils
# import torch.utils.data

import numpy as np
import logging
import os
from typing import List, Tuple
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# ignore from matplotlib 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

device = 'cuda:0'

sys.path.insert(0, os.path.abspath('../src/'))

from models.ds1 import DeepSpeech1WithContextFrames
from ctc.ctc_loss_imp import *
from data.librisubset import get_dataset_libri_sampled_loader, get_datapoint_i
from utils.plot import *
from utils.util import *
from loss.loss  import *

# %%
torch.autograd.set_detect_anomaly(True)

# %%

def get_device_net(FLAGS):
    device = 'cuda:0'
    net = DeepSpeech1WithContextFrames(FLAGS.n_context, FLAGS.drop_prob).to(device)
    return device, net

# create a optimization loop function
def optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, net, 
                       dldw_targets , params_to_match, targets, loss_func, FLAGS):

    i=0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    mae_history = []
    stop_condition = False
    lr_history = []
    while i < FLAGS.max_iter or not stop_condition:
        # x_param_full= torch.concat([x_param, x_pad], dim=2)
        out = net(x_param) # 1 176 29
        out = out.log_softmax(-1)
        # mloss, dldw_f = meta_loss(output, targets, output_sizes, target_sizes, dldw_target,  weight_param)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets,  params_to_match, loss_func)
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0])
        gm_bias_distance   = grad_distance(dldws[1], dldw_targets[1])

        # logging.debug('DLDW target WEIGHT NORM:', torch.norm(dldw_targets[0])) 

        # regloss = tv_norm(x_param)
        if FLAGS.reg == 'L2':
            regloss = torch.norm(x_param, p=2)
        elif FLAGS.reg == 'L1':
            pass
        elif FLAGS.reg == 'TV':
            # need to make x_param from [n_frame, batch size, n_features] to [batch size, 1, n_features, n_frame]
            regloss = tv_norm(x_param.permute(1,0,2).unsqueeze(1))
        else:
            regloss = torch.tensor(0.0)
       
        loss = (1-FLAGS.reg_weight)* mloss + FLAGS.reg_weight * regloss



        optimizer.zero_grad()
        loss.backward()
        grad = x_param.grad.data

        # torch.nn.utils.clip_grad_norm_(x_param, 1.0)
        optimizer.step()
        scheduler.step()

        ## PROJECT NON NEGATIVE
        # x_param = x_param.clamp(min=0)
        # with torch.no_grad():
        #x_param.data = torch.clamp( x_param.data, min=0)


        loss_history.append(loss.item())
        loss_gm_history.append(mloss.item() )
        loss_reg_history.append(regloss.item() )
        lr_history.append(optimizer.param_groups[0]["lr"])

        mae = mae_tensor(x_param, inputs)
        mae_history.append(mae)

        if i % 10 == 0:
            logging.info('Iter, Loss (All-Ggrad-Gweight-Gbias-R), Gradient Norm, Learning Rate, mae: {:4d}, [{:.8f}, {:.4f}, {:.4f}, {:.4f}], {:.4f}, {:.4f}, {:.4f}, {:.4f}'\
                        .format(i, loss.item(), mloss.item(),  gm_weight_distance.item(), gm_bias_distance.item(), regloss.item()
                                    , grad.norm().item(), optimizer.param_groups[0]["lr"], mae)
            )
            # scheduler.step(mloss.item())

        if i % 100 == 0:
            # logging.info('plotted')
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,lr_history, i, FLAGS)
            # import ipdb;ipdb.set_trace()
            pass
            
        
        i+=1
        # stet stop condition true if loss not decrease in last 100 iteration
        if i>100 and loss_history[-1] > min(loss_history[-100:]):
            stop_condition = True
        else:
            stop_condition
    # save the reconstructed x_param, remember to detach and cpu it..
    save_path = os.path.join(FLAGS.exp_path, 'x_param_last.pt')
    torch.save(x_param.detach().cpu(), save_path)

    return x_param

# optimization_loop(x_param, optimizer, scheduler, net, dldw_target, weight_param, targets)

# plot_mfcc(inputs.cpu().squeeze())

# plot_mfcc(x_param.cpu().detach().squeeze())

# plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i)



## write a main entry point for a python script file
# that has args
# 1. choose what is the index of the data point to reconstruct
# 2. choose the learning rate
# 3. choose what kind or regularization (L1, L2, TV)
# 4. choose the weight of that regularization [0,1]
# 5. choose the number of iterations
# 6. choose number of seeds to try
# python3 main.py --index 0 --lr 0.1 --reg L2 --reg_weight 0.05 --iterations 10000 --seeds 5
# example of calling the main function with all args name
# main(index=0, lr=0.1, reg='L2', reg_weight=0.05, iterations=1000, n_seeds=5)        


def main(FLAGS):
    """
    Main function for reconstructing data points with specified hyperparameters.
   
    Parameters:
    - index: Index of the data point to reconstruct.
    - lr: Learning rate for optimization.
    - reg: Type of regularization ('L1', 'L2', 'TV').
    - reg_weight: Weight of the regularization term.
    - iterations: Number of iterations for the optimization.
    """
    # Change all print statements to logging statements
    logging.info('Reconstructing data point at index: {}'.format(FLAGS.index))
    logging.info('Optimizer: {}'.format(FLAGS.optimizer_name))
    logging.info('Learning rate: {}'.format(FLAGS.lr))
    logging.info('Regularization: {}'.format(FLAGS.reg))
    logging.info('Regularization weight: {}'.format(FLAGS.reg_weight))
    logging.info('Number of iterations: {}'.format(FLAGS.iterations))

    #check if exp_path exists or create
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
        logging.info('exp_path {} created'.format(FLAGS.exp_path))
    if not os.path.exists(os.path.join(FLAGS.exp_path, 'figures')):
        os.makedirs(os.path.join(FLAGS.exp_path, 'figures'))    
    logging.info('logging experiment to {}'.format(FLAGS.exp_path))

    device, net = get_device_net(FLAGS)
    logging.info('Device: {}'.format(device))
    logging.info('Network: {}'.format((net.__class__.__name__)))


    # check if example/net_params.pt exists, if not create by saving the net state_dict, if yes then load
    # if not os.path.exists(os.path.join(FLAGS.exp_path, 'net_params.pt')):
    #     torch.save(net.state_dict(), os.path.join(FLAGS.exp_path, 'net_params.pt'))
    #     logging.info('net_params.pt created')
    # else:
    #     net.load_state_dict(torch.load(os.path.join(FLAGS.exp_path, 'net_params.pt')))
    #     logging.info('net_params.pt loaded')

     # get device net dataset loader datapoint i
    # if FLAGS.index != 0:
    #     raise ValueError('script now run for index 0 only')

    # check if input.pt exists, if not create by loading the next_item and save it
    # if not os.path.exists(os.path.join(FLAGS.exp_path, 'input.pt')):
    #     dataset, loader = get_dataset_loader(net)
    #     next_item = get_datapoint_i(iter(loader), 0)
    #     torch.save(next_item, os.path.join(FLAGS.exp_path, 'input.pt'))
    #     logging.info('input.pt created')
    # else:
    #     next_item = torch.load(os.path.join(FLAGS.exp_path, 'input.pt'))
    #     logging.info('input.pt loaded')
    dataset, loader = get_dataset_libri_sampled_loader(net, FLAGS)
    next_item = get_datapoint_i(iter(loader), FLAGS.index)

    logging.info('')


    inputs ,input_sizes = next_item[0]
    logging.info('inputs mean and std: {}, {}'.format(inputs.mean(), inputs.std()))
    # input_sizes is tensor list of inputs.shape[1] elements with value inputs.shape[0]

    targets = next_item[1]
    text = ''.join(net.ALPHABET.get_symbols(targets[0].tolist()))
    logging.info('TEXT: {}'.format(text))
    target_sizes = torch.Tensor([len(t) for t in targets]).int()

    #target is list of tensor with different length, pad it to the same length in a tensor
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    # transfer the data to the GPU
    inputs = inputs.to(device)
    targets = targets.long().to(device)

    input_sizes = input_sizes.long().to(device)
    target_sizes = target_sizes.long().to(device)

    # get the target gradient
    # param to match, a list of pointer to params
    params_to_match = [net.network.out.module[0].weight, net.network.out.module[0].bias]
    out = net(inputs)
    output_sizes = (torch.ones(out.shape[1]) * out.shape[0]).int()
    # output_sizes = torch.Tensor([out.size(0)]).int().to(device)
    out =  out.log_softmax(-1)

    # loss_func = lambda x,y :batched_ctc_logspace_scale(x, y, output_sizes, target_sizes)
    # loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes)
    # loss_func = lambda x,y : batched_ctc_logspace(x, y, output_sizes, target_sizes)
    # loss_func = lambda x,y : batched_ctc_logspace(x, y, output_sizes, target_sizes)

    # loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)
    loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)


    loss_func_lib   = torch.nn.CTCLoss()
    loss = loss_func(out, targets)
    loss_lib = loss_func_lib(out.cpu(), targets.cpu(), output_sizes.cpu(), target_sizes.cpu())
    logging.debug('loss: {}'.format(loss.item()))
    logging.debug('loss by pt lib: {}'.format(loss_lib.item()))
    dldw_targets = torch.autograd.grad(loss, params_to_match)

    ## zero out small values keep 10% largest dldw_target
    # logging.info('zero out small values keep 10% largest dldw_target')
    # dldw_target = dldw_target * (dldw_target.abs() > dldw_target.abs().topk(int(0.1*dldw_target.numel()))[0][-1])
    for ip, p in enumerate(params_to_match):
        p.requires_grad = True
        logging.debug('matching {}. params with shape {} and norm {} first ten {}'.format(ip, p.shape, p.norm(), p.flatten()[:10]))
        logging.debug('                    gradient norm {}'.format(dldw_targets[ip].norm()))

    # init x_param
    torch.manual_seed(0)

    x_init =  init_a_point(inputs, FLAGS)
    x_param = torch.nn.Parameter(x_init.to(device),requires_grad=True)


    if FLAGS.optimizer_name.lower() == 'adam':
        optimizer = optim.Adam([x_param], lr=FLAGS.lr)
    elif FLAGS.optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD([x_param], lr=FLAGS.lr)
    else:
        raise ValueError(f"Unknown optimizer: {FLAGS.optimizer_name}")

    # reduce lr at epoch 250, 500, 750 half
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(250,2000,250)), gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience=50)


    # suggest an experiment name base on datapoint index, optimizer name,  learning rate, regularizer, regularizer weight
    logging.info('Experiment Name: {}'.format(os.path.basename(FLAGS.exp_path)))

    optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, net,
                       dldw_targets = dldw_targets, params_to_match =  params_to_match, targets = targets, loss_func=loss_func,  FLAGS= FLAGS)


## main


FLAGS = argparse.Namespace(index=2, optimizer_name='Adam', lr=0.5, reg='None', reg_weight=0, iterations=10000, n_seeds=5, max_iter=10000,
                            n_context=6, drop_prob=0, init_method='uniform', batch_min_dur=1000, batch_max_dur=2000, batch_size=1)

exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/example_v2/'
exp_name = f"idx_{FLAGS.index}_init_{FLAGS.init_method}_opt_{FLAGS.optimizer_name}_lr_{FLAGS.lr}_reg_{FLAGS.reg}_regw_{FLAGS.reg_weight}"
FLAGS.exp_path=os.path.join(exp_path, exp_name)

main(FLAGS)

# %% [markdown]
# 

# %%




