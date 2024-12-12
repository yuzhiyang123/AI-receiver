# Dataset importation & Interface of channel mapping model
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset
from model.mixer import Mapping_Net


def get_dataset(ratio: list, freqs: list, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='.'):
    assert sum(ratio) == 1
    print(freqs)
    freqs2 = []
    for i in range(64):
         if i not in freqs:
            freqs2.append(i)
    print(freqs2)

    channel_file1 = dataset_path + name + '/data.npy'
    print(channel_file1)

    channel = np.load(channel_file1)  # N*1*ant*car complex
    channel = torch.tensor(channel)
    print(channel.shape)
    channel = rearrange(channel, 'b c ant car -> b ant car c')
    channel_torch = torch.cat((channel.real, channel.imag), 3)  # N*ant*car*2 float tensor
    num_data = channel_torch.shape[0]

    # Normalization
    channel_torch = channel_torch * 1e5

    channel_P = channel_torch[:, :, freqs, :]
    channel_H = channel_torch[:, :, freqs2, :]

    torch.manual_seed(seed)
    perm = torch.randperm(num_data)

    num = int(0.8 * num_data)
    ids = perm[0:num]
    channel_P_train = channel_P[ids]
    channel_H_train = channel_H[ids]
    # torch.save(channel_P_train, "channel_P_train_"+name+".pt")
    # torch.save(channel_H_train, "channel_H_train_"+name+".pt")
    ids = perm[num:]
    channel_P_test = channel_P[ids]
    channel_H_test = channel_H[ids]
    # torch.save(channel_P_test, "channel_P_test_"+name+".pt")
    # torch.save(channel_H_test, "channel_H_test_"+name+".pt")
    # channel_P_train = torch.load("channel_P_train_"+name+".pt")
    # channel_H_train = torch.load("channel_H_train_"+name+".pt")
    # channel_P_test = torch.load("channel_P_test_"+name+".pt")
    # channel_H_test = torch.load("channel_H_test_"+name+".pt")
    datasets = [TensorDataset(channel_P_train, channel_H_train),
                TensorDataset(channel_P_test, channel_H_test)]
    #     datasets.append(dataset)
    # for r in ratio:
    #     num = int(r * num_data)
    #     ids = perm[cursor:cursor+num]
    #     dataset = TensorDataset(channel_torch[ids])
    #     cursor = cursor + num
    #     datasets.append(dataset)
    return datasets  # A list of datasets


def get_mixer_net(input_ant_size, input_car_size, ant_size, car_size, depth):
    return Mapping_Net(input_ant_size, input_car_size, ant_size, car_size, depth)


def Nmse(input, output):
    num = input.shape[0]
    err = (input - output).view(num, -1)
    nmse = torch.sum(err.pow(2), dim=1).div(torch.sum(input.view(num, -1).pow(2), dim=1))
    return torch.mean(nmse).item()

