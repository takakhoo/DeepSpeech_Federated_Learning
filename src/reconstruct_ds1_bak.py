# %%
import torch
import torch.optim as optim
import datetime
import argparse
from torch import nn
import sys, os

# Add the 'modules/deepspeech/src/' directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules/deepspeech/src')))
import deepspeech

# from deepspeech.networks.utils import OverLastDim
from deepspeech.data import preprocess
from torchvision.transforms import Compose
from deepspeech.data.loader import collate_input_sequences
from ctc.ctc_loss_imp import ctc_loss_imp
import torch.utils
import torch.utils.data
from deepspeech.data.datasets.librispeech import LibriSpeech

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os
from typing import List, Tuple
from models.ds1 import DeepSpeech1WithContextFrames
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# ignore from matplotlib 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from IPython.display import display

from utils.plot import *
from utils.util import *
from loss.loss  import *
device = 'cuda:0'


def get_device_net(FLAGS):
    device = 'cuda:0'
    net = DeepSpeech1WithContextFrames(FLAGS.n_context, FLAGS.drop_prob).to(device)
    return device, net

def get_dataset_loader(net):
    dataset = LibriSpeech(root='/scratch/f006pq6/datasets/librispeech/', subsets=['test-clean'], download=True,
                          transform=net.transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=collate_input_sequences,
                                         pin_memory=torch.cuda.is_available(),
                                         num_workers=4,
                                         batch_size=1,
                                         shuffle=False)
    print('number of utterances:', len(dataset))
    print('example shape of input:', dataset[0][0][0].shape)
    print('number of frames in example:', dataset[0][0][1])
    print('example target:', dataset[0][1])
    return dataset, loader

def get_datapoint_i(loader_iterator, idx):
    for i in range(idx):
        next(loader_iterator)
    next_item = next(loader_iterator)
    print('next item shape of input:', next_item[0][0].shape)
    print('next item number of frames:', next_item[0][1])
    print('next item target:', next_item[1])
    return next_item



# ------------------------------------------------------------------------------
# Meta loss
# ------------------------------------------------------------------------------
def meta_loss(output, targets, output_sizes, target_sizes, dldw_targets,  params_to_match, loss_func):
    loss = loss_func(output, targets)
    dldws = torch.autograd.grad(loss, params_to_match, create_graph=True)
    # loss = ((dldw-dldw_target)**2).mean() #MSE
    #loss = 1 - torch.nn.functional.cosine_similarity(dldw.reshape(1,-1), dldw_target.reshape(1,-1))    
    loss = 0 
    for dldw, dldw_target in zip(dldws, dldw_targets):
        #loss += torch.nn.functional.mse_loss(dldw, dldw_target)
        # for each dldw and dldw_targets pair, only compute grad_distance for top 50% abs value of dldw_targets


        loss += grad_distance(dldw, dldw_target, FLAGS)

    return loss,dldws


def init_a_point(inputs, FLAGS):
    if FLAGS.init_method == 'uniform':
        # init x_init varialbe with unifrom [-1,1]
        logging.info('init with uniform')
        x_init = torch.rand_like(inputs) * 2 - 1
    elif FLAGS.init_method == 'normal':
        # init x_init varialbe with normal distribution
        logging.info('init with normal')
        x_init = torch.randn_like(inputs)
    elif FLAGS.init_method == 'same':
        # init x_init varialbe with same as inputs
        logging.info('init with same') 
        x_init = inputs.clone()
    elif FLAGS.init_method == 'same_noisy':
        # init x_init varialbe with same as inputs + noise
        logging.info('init with same_noisy')
        x_init = inputs.clone() + torch.randn_like(inputs) * 0.01

    logging.info('init mean, std:{} {}'.format(x_init.mean(), x_init.std()))
    x_param = torch.nn.Parameter(x_init.to(device),requires_grad=True)
    # x_param_full = torch.concat([x_param, x_pad], dim=2)
    return x_param

def zero_order_optimization_loop(inputs, x_param, output_sizes, target_size,
                                 net,
                                    dldw_targets , params_to_match, targets,  FLAGS):
    net.eval()
    loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_size,reduction='mean')
    

    i = 0 
    stop_condition = False

    def get_meta_loss(x_param):
        out = net(x_param)
        out = out.log_softmax(-1)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets, params_to_match, loss_func)
        return mloss, dldws
    
    tolerance = 10
    step_size = FLAGS.zero_lr

    loss_history = []
    loss_gm_history = []
    loss_reg_history = []

    while i < FLAGS.zero_max_iter or not stop_condition:
        # random 16 directions in the space of x_param
        directions = torch.randn(8, *x_param.shape).to(device)
        # normalize direction so that they have unit length
        shape = directions.shape
        directions = directions.reshape(8,-1)
        directions = torch.functional.F.normalize(directions, dim=1)
        directions = directions.reshape(shape)
        # create a list to store the loss for each direction
        losses = []
        current_loss, _ = get_meta_loss(x_param)
        # logging.info('Current loss: {}'.format(current_loss.item()))

        for d in directions:
            x_param_new = x_param + step_size * d
            mloss, _ = get_meta_loss(x_param_new)
            losses.append(mloss.item())

        # find the best direction by averaging direction that reduce loss
        # best_direction = directions[torch.tensor(losses) < current_loss.item()]
        # find the best direction by the smallest loss, argmin
        best_direction = directions[np.argmin(losses)]

        if  np.min(losses) < current_loss.item():
            # best_direction = best_direction.mean(dim=0)
            x_param = x_param + step_size * best_direction
            tolerance = 10
            step_size = FLAGS.zero_lr
        else:  
            logging.info('No direction found, reducing step size, tolerance: {}, {}'.format(step_size,tolerance))
            tolerance -=1
            step_size *= 0.5
            if tolerance < 0:
                stop_condition = True

        mae = torch.mean(torch.abs(x_param - inputs))
        logging.info('iter {}  loss: {}, step size: {}, mae: {}'.format(i, np.min(losses), step_size, mae.item()))


        loss_history.append(current_loss.item())    
        loss_gm_history.append(0)
        loss_reg_history.append(0)
        if i % 20 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i, FLAGS, prefix='zero_order_')
            pass
 
        i += 1

    return x_param

def first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes,
                                  optimizer, scheduler, net,
                                  dldw_targets , params_to_match, targets,  FLAGS):
    loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    i=0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    stop_condition = False
    while i < FLAGS.max_iter or not stop_condition:
        # x_param_full= torch.concat([x_param, x_pad], dim=2)
        out = net(x_param) # 1 176 29
        out = out.log_softmax(-1)
        # mloss, dldw_f = meta_loss(output, targets, output_sizes, target_sizes, dldw_target,  weight_param)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets,  params_to_match, loss_func)
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], FLAGS)
        gm_bias_distance   = grad_distance(dldws[1], dldw_targets[1], FLAGS)

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


        mae = torch.mean(torch.abs(x_param - inputs))

        loss_history.append(loss.item())
        loss_gm_history.append(mloss.item() )
        loss_reg_history.append(regloss.item() )

        if i % 10 == 0:
            logging.info('Iter, Loss (A-G-Gw-Gb-R), Gradient Norm, Learning Rate, MAE: {:4d}, {:.8f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'\
                        .format(i, loss.item(), mloss.item(),  gm_weight_distance.item(), gm_bias_distance.item(), regloss.item()
            , grad.norm().item(), optimizer.param_groups[0]["lr"], mae.item()))
            # scheduler.step(mloss.item())

        if i % 100 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i, FLAGS)
            pass
            
        
        i+=1
        # stet stop condition true if loss not decrease in last 100 iteration
        if i>100 and loss_history[-1] > min(loss_history[-100:]):
            stop_condition = True
        else:
            stop_condition
    return x_param


# create a optimization loop function
def optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, net, 
                       dldw_targets , params_to_match, targets,  FLAGS):

    # loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    if not os.path.exists(os.path.join(FLAGS.exp_path, 'x_param_first_order.pt')):
        logging.info('Running first order optimization loop')
        x_param = first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, optimizer, scheduler, net, dldw_targets, params_to_match, targets, FLAGS) 
        torch.save(x_param.detach().cpu(), os.path.join(FLAGS.exp_path, 'x_param_first_order.pt'))
        logging.info('x_param_first_order.pt saved')
    else:
        x_param = torch.load(os.path.join(FLAGS.exp_path, 'x_param_first_order.pt')).to(device)
        logging.info('x_param_first_order.pt loaded')


    logging.info('Running zero order optimization loop')
    x_param = zero_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, net, dldw_targets, params_to_match, targets, FLAGS)


    save_path = os.path.join(FLAGS.exp_path, 'x_param_last.pt')
    torch.save(x_param.detach().cpu(), save_path)

    return x_param


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
    logging.info('Number of iterations: {}'.format(FLAGS.max_iter))

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


    # if exist FLAGS.cpt_resume load the checkpoint
    if FLAGS.cpt_resume is not None:
        state_dict = torch.load(FLAGS.cpt_resume)['network']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = 'network.' + k  # Add 'network.' prefix
            new_state_dict[new_key] = v
        net.load_state_dict(new_state_dict)
        logging.info('Checkpoint loaded from {}'.format(FLAGS.cpt_resume))
    else:
        # loging random init weight
        logging.info('Random init weight')


    # if not os.path.exists(os.path.join(FLAGS.exp_path, 'net_params.pt')):
    #     torch.save(net.state_dict(), os.path.join(FLAGS.exp_path, 'net_params.pt'))
    #     logging.info('net_params.pt created')
    # else:
    #     net.load_state_dict(torch.load(os.path.join(FLAGS.exp_path, 'net_params.pt')))
    #     logging.info('net_params.pt loaded')

     # get device net dataset loader datapoint i
    if FLAGS.index != 0:
        raise ValueError('script now run for index 0 only')

    # check if input.pt exists, if not create by loading the next_item and save it
    if not os.path.exists(os.path.join(FLAGS.exp_path, 'input.pt')):
        dataset, loader = get_dataset_loader(net)
        next_item = get_datapoint_i(iter(loader), 0)
        torch.save(next_item, os.path.join(FLAGS.exp_path, 'input.pt'))
        logging.info('input.pt created')
    else:
        next_item = torch.load(os.path.join(FLAGS.exp_path, 'input.pt'))
        logging.info('input.pt loaded')
    
    logging.info('')

    target_transform = Compose([str.lower,
                        net.ALPHABET.get_indices,
                        torch.IntTensor])


    inputs = next_item[0][0]
    logging.info('inputs mean and std: {}, {}'.format(inputs.mean(), inputs.std()))
    input_sizes = torch.Tensor([inputs.shape[0]]).int()
    targets = target_transform(next_item[1][0])
    target_sizes = torch.Tensor([len(targets)]).int()

    # transfer the data to the GPU
    inputs = inputs.to(device)
    input_sizes = input_sizes.to(device)
    targets = targets.to(device)
    target_sizes = target_sizes.to(device)

    # get the target gradient
    # param to match, a list of pointer to params
    params_to_match = [net.network.out.module[0].weight, net.network.out.module[0].bias]
    out = net(inputs)
    output_sizes = torch.Tensor([out.size(0)]).int().to(device)
    out =  out.log_softmax(-1)
    loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')
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

    x_init = init_a_point(inputs, FLAGS)
    # x_init = torch.randn_like(inputs).to(device)
    x_param = torch.nn.Parameter(x_init.to(device), requires_grad=True)
    logging.debug('init mean, std: {}, {}'.format(x_init.mean(), x_init.std()))

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
                       dldw_targets = dldw_targets, params_to_match =  params_to_match, targets = targets,  FLAGS= FLAGS)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Reconstruct a data point with specified parameters.")

    # # Add arguments
    # parser.add_argument("--index", type=int, required=True, help="Index of the data point to reconstruct")
    # parser.add_argument("--optimizer", type=str, required=True, help="Optimizer to use for optimization")
    # parser.add_argument("--lr", type=float, required=True, help="Learning rate for optimization")
    # parser.add_argument("--reg", type=str, required=True, choices=["L1", "L2", "TV", "None"], help="Type of regularization")
    # parser.add_argument("--reg_weight", type=float, required=True, help="Weight of the regularization term")
    # parser.add_argument("--iterations", type=int, required=True, help="Number of iterations for the optimization")
    # parser.add_argument("--seeds", type=int, required=True, help="Number of seeds to try (unused)")  

    # args = parser.parse_args()
    # FLAGS = vars(args)
    # # Call the main function with the parsed arguments

    FLAGS = argparse.Namespace(index=0, optimizer_name='Adam', lr=0.5, zero_lr=100, reg='None', reg_weight=0.0,  n_seeds=10, max_iter=2000, zero_max_iter=200,
                               n_context=6, drop_prob=0.0, init_method='uniform', cpt_resume='/scratch/f006pq6/projects/deepspeech-myrtle/exp/ds1-1.pt',
                               top_grad_percentage=1.0)

    exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/example_v2/'
    # get name of the ckp file
    cpt_name = os.path.basename(FLAGS.cpt_resume) if FLAGS.cpt_resume is not None else 'None'
    exp_name = f"idx_{FLAGS.index}_init_{FLAGS.init_method}_opt_{FLAGS.optimizer_name}_lr_{FLAGS.lr}_reg_{FLAGS.reg}_regw_{FLAGS.reg_weight}_top-grad-perc_{FLAGS.top_grad_percentage}_cpt_{cpt_name}"
    FLAGS.exp_path=os.path.join(exp_path, exp_name)

    main(FLAGS)
