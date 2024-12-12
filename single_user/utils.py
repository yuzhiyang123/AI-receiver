import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from einops import rearrange


def get_dataset(ratio: list, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='.'):
    dest_train = "channel_train_"+name+".pt"
    dest_test = "channel_test_"+name+".pt"
    dest_val = "channel_val_" + name + ".pt"  # not always exist

    if os.path.exists(dest_train) and os.path.exists(dest_test):
        channel_train = torch.load(dest_train)
        print(torch.mean(channel_train*channel_train))
        channel_test = torch.load(dest_test)
        if os.path.exists(dest_val):
            channel_val = torch.load(dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]
    else:
        channel_file = dataset_path + name + '/data.npy'
        print("Loading channels", channel_file)

        channel = np.load(channel_file)  # N*1*ant*car complex
        channel = torch.tensor(channel)
        # print(channel.shape)
        channel = rearrange(channel, 'b c ant car -> b ant car c')
        channel_torch = torch.cat((channel.real, channel.imag), 3)  # N*ant*car*2 float tensor
        num_data = channel_torch.shape[0]

        # Normalization
        channel_torch = channel_torch * 1e5

        torch.manual_seed(seed)
        perm = torch.randperm(num_data)

        num = int(ratio[0] * num_data)
        ids = perm[0:num]
        channel_train = channel_torch[ids]
        torch.save(channel_train, dest_train)

        num2 = int(ratio[1] * num_data)
        ids = perm[num:num+num2]
        channel_test = channel_torch[ids]
        torch.save(channel_test,  dest_test)

        if len(ratio) == 3:
            num3 = int(ratio[2] * num_data)
            ids = perm[num+num2:num+num2+num3]
            channel_val = channel_torch[ids]
            torch.save(channel_val,  dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]


def NMSE(label, output):
    num = label.shape[0]
    err = (label - output).view(num, -1)
    nmse = torch.sum(err.pow(2), dim=1).div(torch.sum(label.view(num, -1).pow(2), dim=1))
    return torch.mean(nmse).item()


def cosine_similarity(label, output):
    label_r = label[:, :, :, 0]
    label_i = label[:, :, :, 1]
    output_r = output[:, :, :, 0]
    output_i = output[:, :, :, 1]
    x_r = label_r * output_r + label_i * output_i
    x_i = label_r * output_i - label_i * output_r
    cos_sim = x_r.sum(dim=1).pow(2) + x_i.sum(dim=1).pow(2)
    #print(torch.sum(x_r), torch.sum(x_i.pow(2)))
    #print(torch.sum(label.pow(2)), torch.sum(output.pow(2)))
    #cos_sim = torch.sum(x_r.pow(2), dim=1) + torch.sum(x_i.pow(2), dim=1)
    ener = torch.sum(label.pow(2), dim=(1, 3)) * torch.sum(output.pow(2), dim=(1, 3))
    return torch.mean(torch.sqrt(cos_sim.div(ener))).item()
