import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import sys
import os

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


def Sampling(num_samples,time_all,time_pt,data_train,sigma,device):
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] 
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples
