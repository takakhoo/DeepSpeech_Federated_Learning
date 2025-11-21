# %%
import torch
import torch.optim as optim
import argparse
from torch import nn
import sys, os
import time

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

sys.path.insert(0, os.path.abspath('../src/'))

from models.ds2 import DeepSpeech2

from ctc.ctc_loss_imp import *
from data.librisubset import *
from utils.plot import *
from utils.util import *
from loss.loss  import *

# %%
def get_device_net(FLAGS, use_relu):
    device = 'cuda:0'
    net = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    return device, net

def zero_order_optimization_loop(inputs, x_param, output_sizes, target_size,
                                 net,
                                    dldw_targets , params_to_match, targets, prefix, FLAGS):
    device = inputs.device
    net.eval()

    loss_func = lambda x,y :batched_ctc_v2(x, y, output_sizes, target_size)
    

    i = 0 
    stop_condition = False

    def get_meta_loss(x_param):
        out = net(x_param)
        out = out.log_softmax(-1)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets, params_to_match, loss_func, FLAGS)
        return mloss, dldws
    
    tolerance = 10
    step_size = FLAGS.zero_lr

    loss_history = []
    loss_gm_history = []
    loss_reg_history = []

    while i < FLAGS.zero_max_iter and not stop_condition:
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
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i, prefix=prefix, FLAGS=FLAGS)
            pass
 
        i += 1

    return x_param

def first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes,
                                  optimizer, scheduler, net,
                                  dldw_targets , params_to_match, targets,prefix,  FLAGS):
    net.train()
    loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    i=0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    stop_condition = False
    while i < FLAGS.max_iter and not stop_condition:
        # x_param_full= torch.concat([x_param, x_pad], dim=2)
        out = net(x_param) # 1 176 29
        out = out.log_softmax(-1)
        # mloss, dldw_f = meta_loss(output, targets, output_sizes, target_sizes, dldw_target,  weight_param)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets,  params_to_match, loss_func, FLAGS)
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], FLAGS)
        gm_bias_distance   =  0.0 #grad_distance(dldws[1], dldw_targets[1], FLAGS)

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
                        .format(i, loss.item(), mloss.item(),  gm_weight_distance.item(), gm_bias_distance, regloss.item()
            , grad.norm().item(), optimizer.param_groups[0]["lr"], mae.item()))
            # scheduler.step(mloss.item())

        if i % 100 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i,prefix=prefix, FLAGS=FLAGS)
            pass
            
        
        i+=1
        # stet stop condition true if loss not decrease in last 100 iteration
        if i>100 and loss_history[-1] > min(loss_history[-100:]):
            stop_condition = True
        else:
            stop_condition
    return x_param


# ---------------------------------------------------------------------------- #
#                      create a optimization loop function                     #
# ---------------------------------------------------------------------------- #
def optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, net, 
                       dldw_targets , params_to_match, targets,prefix='',  FLAGS=None):

    device = inputs.device

    # loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    if not os.path.exists(os.path.join(FLAGS.exp_path, prefix+'_x_param_first_order.pt')):
        logging.info('Running first order optimization loop')
        x_param = first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, optimizer, scheduler, net, dldw_targets, params_to_match, targets,prefix+'_firstorder', FLAGS) 
        torch.save(x_param.detach().cpu(), os.path.join(FLAGS.exp_path, prefix+'_x_param_first_order.pt'))
        logging.info('x_param_first_order.pt saved')
    else:
        x_param = torch.load(os.path.join(FLAGS.exp_path, prefix+'_x_param_first_order.pt')).to(device)
        logging.info('x_param_first_order.pt loaded')


    logging.info('Running zero order optimization loop')
    x_param = zero_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, net, dldw_targets, params_to_match, targets,prefix+'_zeroorder',FLAGS)


    return x_param

# ---------------------------------------------------------------------------- #
#                 Reconstruct all datapoint in a torch dataset                 #
# ---------------------------------------------------------------------------- #
def reconstruct_dataset(network, device, dataloader, FLAGS):
    torch.manual_seed(0)

    # loop through item in the dataloader
    for (i, batch) in enumerate(dataloader):

        # batch = A tuple of `((batch_x, batch_out_lens), batch_y)` where:
        logging.info('#'*20)
        logging.info('Processing batch {}/{}'.format(i, len(dataloader)))

        inputs ,input_sizes = batch[0]
        logging.info('inputs mean and std: {}, {}'.format(inputs.mean(), inputs.std()))
        # input_sizes is tensor list of inputs.shape[1] elements with value inputs.shape[0]

        targets = batch[1]
        text = ''.join(network.ALPHABET.get_symbols(targets[0].tolist()))
        logging.info('TEXT: {}'.format(text))
        target_sizes = torch.Tensor([len(t) for t in targets]).int()

        #target is list of tensor with different length, pad it to the same length in a tensor
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)


        # transfer the data to the GPU
        inputs = inputs.to(device)
        targets = targets.long().to(device)

        input_sizes = input_sizes.long().to(device)
        target_sizes = target_sizes.long().to(device)


        out = network(inputs)


        # params_to_match = [network.network.fc.module[1].weight, network.network.out.module[0].bias]
        params_to_match = [network.network.fc.module[1].weight] # deep speech 2 doesnt use bias in last FC
        output_sizes = (torch.ones(out.shape[1]) * out.shape[0]).int()
        out =  out.log_softmax(-1)

        loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)

        loss = loss_func(out, targets)
        # loss_func_lib   = torch.nn.CTCLoss()
        # loss_lib = loss_func_lib(out.cpu(), targets.cpu(), output_sizes.cpu(), target_sizes.cpu())

        logging.debug('loss: {}'.format(loss.item()))
        dldw_targets = torch.autograd.grad(loss, params_to_match)

        ## zero out small values keep 10% largest dldw_target
        # logging.info('zero out small values keep 10% largest dldw_target')
        # dldw_target = dldw_target * (dldw_target.abs() > dldw_target.abs().topk(int(0.1*dldw_target.numel()))[0][-1])
        for ip, p in enumerate(params_to_match):
            p.requires_grad = True
            logging.debug('matching {}. params with shape {} and norm {} first ten {}'.format(ip, p.shape, p.norm(), p.flatten()[:10]))
            logging.debug('                    gradient norm {}'.format(dldw_targets[ip].norm()))


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

        # timing the optimization loop
        start_time = time.time()
        x_param = optimization_loop(inputs, x_param, output_sizes, target_sizes,
                        optimizer, scheduler, network,
                        dldw_targets = dldw_targets, params_to_match =  params_to_match, targets = targets, prefix=f'sampleidx_{i}', FLAGS=FLAGS)
        end_time = time.time()
                        
        save_path = os.path.join(FLAGS.exp_path, f'sampleidx_{i}_' + 'x_param_last.pt'.format(i))
        # save x_param, optimization time, inputs, targets
        torch.save({ 'x_param': x_param.detach().cpu(),
                    'time': end_time - start_time,
                    'inputs': inputs.detach().cpu(), 
                    'targets': targets.detach().cpu(),
                    'transcript': text
                    }, save_path)

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

    # ---------------------------------------------------------------------------- #
    #                      Loading network and devices.......                      #
    # ---------------------------------------------------------------------------- #
    device, net = get_device_net(FLAGS, use_relu=False)
    logging.info('Device: {}'.format(device))
    logging.info('Network: {}'.format((net.__class__.__name__)))

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

    # ---------------------------------------------------------------------------- #
    #                     Loading dataset ....................                     #
    # ---------------------------------------------------------------------------- #

    dataset, loader = get_dataset_libri_sampled_folder_subset(net, FLAGS)

    reconstruct_dataset(net, device, loader, FLAGS)


#rsync -avz lisplab0:/scratch/f006pq6/datasets/librispeech_sampled_600_file_0s_4s . define function to pretty print all the configuration in argparse, column tab
def pretty_print_config(FLAGS):
    for key, value in FLAGS.__dict__.items():
        print("{:30} {}".format(key, value))




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

    # ckp_path = '/scratch/f006pq6/projects/deepspeech-myrtle/exp/ds1-1.pt'

    parser = argparse.ArgumentParser(description="Reconstruct a data point with specified parameters.")

    # Add arguments with default values
    # parser.add_argument("--index", type=int, required=True, help="Index of the data point to reconstruct")
    parser.add_argument("--batch-start", required=True,type=int, default='Adam', help="index of the start")
    parser.add_argument("--batch-end",   required=True,type=int, default='Adam', help="index of the end")

    parser.add_argument("--optimizer_name", type=str, default='Adam', help="Optimizer to use for optimization")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimization")
    parser.add_argument("--zero_lr", type=float, default=100, help="Learning rate for zero order optimization")
    parser.add_argument("--reg", type=str, default='None', choices=["L1", "L2", "TV", "None"], help="Type of regularization")
    parser.add_argument("--reg_weight", type=float, default=0.0, help="Weight of the regularization term")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to try")
    parser.add_argument("--max_iter", type=int, default=2000, help="Number of iterations for the optimization")
    parser.add_argument("--zero_max_iter", type=int, default=200, help="Number of iterations for zero order optimization")
    parser.add_argument("--n_context", type=int, default=6, help="Number of context frames")
    parser.add_argument("--drop_prob", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--init_method", type=str, default='uniform', help="Initialization method")
    parser.add_argument("--cpt_resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--top_grad_percentage", type=float, default=1.0, help="Top gradient percentage")
    parser.add_argument("--batch_min_dur", type=int, default=0, help="Minimum duration of batch")
    parser.add_argument("--batch_max_dur", type=int, default=1000, help="Maximum duration of batch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--distance_function", default='cosine', type=str)
    parser.add_argument("--dataset_path", type=str, default='/scratch/f006pq6/datasets/librispeech_sampled_600_file_0s_4s', help="Batch size")


    FLAGS = parser.parse_args()

    assert FLAGS.batch_size == 1, "Batch size must be 1"

    if FLAGS.batch_min_dur == 0 and FLAGS.batch_max_dur == 1000:
        exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/0s-1s/'
    elif FLAGS.batch_min_dur == 1000 and FLAGS.batch_max_dur == 2000:
        exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/1s-2s/'
    elif FLAGS.batch_min_dur == 2000 and FLAGS.batch_max_dur == 3000:
        exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/2s-3s/'
    elif FLAGS.batch_min_dur == 3000 and FLAGS.batch_max_dur == 4000:
        exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/3s-4s/'

    # exp_path='/scratch/f006pq6/projects/asr-grad-reconstruction/logging/0s-1s/'
    # get name of the ckp file
    cpt_name = os.path.basename(FLAGS.cpt_resume) if FLAGS.cpt_resume is not None else 'None'
    exp_name = f"DS2_batchstart_{FLAGS.batch_start}_batch_end_{FLAGS.batch_end}_init_{FLAGS.init_method}_opt_{FLAGS.optimizer_name}_lr_{FLAGS.lr}_reg_{FLAGS.reg}_regw_{FLAGS.reg_weight}_top-grad-perc_{FLAGS.top_grad_percentage}_cpt_{cpt_name}"
    FLAGS.exp_path=os.path.join(exp_path, exp_name)

    pretty_print_config(FLAGS)

    main(FLAGS)
