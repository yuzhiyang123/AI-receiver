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


def NMSE(input, output):
    num = input.shape[0]
    err = (input - output).view(num, -1)
    nmse = torch.sum(err.pow(2), dim=1).div(torch.sum(input.view(num, -1).pow(2), dim=1))
    return torch.mean(nmse).item()


MSE = lambda H: torch.mean(H.real * H.real + H.imag * H.imag)


def LS(X, Y, N0, kp, Es=1):
    H = Y @ torch.linalg.pinv(X)
    var = N0 / (Es * kp)
    return H, var


def BER(target, est):
    err_re = est.real.mul(target.real).ge(0)
    err_im = est.imag.mul(target.imag).ge(0)
    err = err_re * err_im
    return 1 - torch.mean(err.float()).item()
    # return 1 - torch.mean(err.float(), dim=[1,2])


def init_mapping_net(model, is_mapping1=True):
    if is_mapping1:
        model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_8ant1.pth'))
    else:
        model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_8ant2.pth'))


def demod_QPSK(u, v):
    def demod(u, v, p1):
        p_1 = p1 / (p1 + (1-p1) * torch.exp(-2 * u.div(v)))
        u_post = 2 * p_1 - 1
        v_post = torch.mean(1 - u_post.pow(2), dim=1, keepdim=True)
        return u_post, v_post

    u_post_r, v_post_r = demod(1.41421356 * u.real, v, 0.5)
    u_post_i, v_post_i = demod(1.41421356 * u.imag, v, 0.5)
    return (u_post_r + 1j * u_post_i)/1.41421356, (v_post_r + v_post_i)/2


def demod_QPSK2(u, gamma):
    u = u * (gamma * 1.41421356)
    re = u.real.tanh().div(1.41421356)
    im = u.imag.tanh().div(1.41421356)
    E = re * re + im * im
    return re + 1j * im, 1 - E

