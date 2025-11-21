import torch
import logging
def tv_norm( x):
    # Compute differences along the y-axis
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    # Compute differences along the x-axis
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    # Compute total variation
    tv = torch.sum(dx) + torch.sum(dy)
    # Scale by the strength parameter
    return tv



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
    # x_param_full = torch.concat([x_param, x_pad], dim=2)
    return x_init