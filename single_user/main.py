import torch
from einops import rearrange
import math
import numpy as np
import csv

from semi_blind import *
from utils import get_dataset


def get_pilot(length, n):
    phi = torch.arange(length).view(1, -1) * torch.arange(n).view(-1, 1)
    phi = (2 * 3.1415926 / length) * phi
    return torch.exp(1j * phi)


def get_channels(num, n_ant, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='32ant_64car_300k', start=0):
    dest_test = "channel_test_" + name + ".pt"
    channel_test = torch.load(dest_test)
    ener = torch.mean(channel_test*channel_test)
    total_data = channel_test.shape[0]
    torch.manual_seed(seed)
    perm = torch.randperm(total_data)
    ids = perm[start:num+start]
    return channel_test[ids, :n_ant, :, :], ener


def get_YHX(SNR_dB, chosen_cars, num, n_ant, cp, cd, Nue, Ndsym, Npsym,
            seed=1234, dataset_path='/mnt/HD2/yyz/MIMOsemiblinddata/'):
    # H = torch.load("channels.pth")
    H, H_ener = get_channels(num*Nue, n_ant, seed, dataset_path)

    H = H[:, :, :, 0] + 1j * H[:, :, :, 1]

    mask = torch.zeros(1, cp + cd, cp + cd)
    j = cp
    k = 0
    for i in range(cp + cd):
        if k < cp and i == chosen_cars[k]:
            mask[0][i][k] = 1
            k += 1
        else:
            mask[0][i][j] = 1
            j += 1
    H = H @ (mask + 0j)
    H = rearrange(H, '(n ue) ant car -> n car ant ue', n=num)

    noise_var = math.pow(10, -SNR_dB / 10) * Nue * H_ener

    constellation = torch.tensor([-1, 1]).view(1, -1) + 1j * torch.tensor([-1, 1]).view(-1, 1)
    constellation = constellation.view(1, 1, 1, -1) * 0.7071178

    X_ori = torch.randint(0, 4, (1, num * cp * Nue * Ndsym))
    Xd1 = torch.zeros(4, num * cp * Nue * Ndsym).scatter_(dim=0, index=X_ori, value=1) + 0j
    Xd1 = constellation @ Xd1
    Xd1 = Xd1.view(num, cp, Nue, Ndsym)

    Xp = get_pilot(Npsym, Nue)
    Xp = Xp.view(1, 1, Nue, Npsym).repeat(num, cp, 1, 1)

    X = torch.cat((Xp, Xd1), dim=3)

    X_ori = torch.randint(0, 4, (1, num * cd * Nue * (Npsym + Ndsym)))
    Xd2 = torch.zeros(4, num * cd * Nue * (Npsym + Ndsym)).scatter_(dim=0, index=X_ori, value=1) + 0j
    Xd2 = constellation @ Xd2
    Xd2 = Xd2.view(num, cd, Nue, (Npsym + Ndsym))
    X = torch.cat((X, Xd2), dim=1)
    Y = H @ X
    Y = Y + math.sqrt(noise_var) * torch.randn_like(Y)
    # print(H_ener, noise_var)

    return Y, H, X, noise_var, H_ener


def get_YHX_simple(SNR_dB, chosen_cars, num, n_ant, cp, cd, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOsemiblinddata/', start=0):
    H, H_ener = get_channels(num, n_ant, seed=seed, start=start)

    H = H[:, :, :, 0] + 1j * H[:, :, :, 1]
    eners = torch.mean(H.real * H.real + H.imag * H.imag, dim=[1, 2]).view(-1, 1, 1, 1)

    mask = torch.zeros(1, cp + cd, cp + cd)
    j = cp
    k = 0
    for i in range(cp + cd):
        if k < cp and i == chosen_cars[k]:
            mask[0][i][k] = 1
            k += 1
        else:
            mask[0][i][j] = 1
            j += 1
    H = H @ (mask + 0j)
    H = rearrange(H, '(n ue) ant car -> n car ant ue', n=num)

    noise_var = math.pow(10, -SNR_dB / 10) * eners

    constellation = torch.tensor([-1, 1]).view(1, -1) + 1j * torch.tensor([-1, 1]).view(-1, 1)
    constellation = constellation.view(1, 1, 1, -1) * 0.7071178

    # X_ori = torch.randint(0, 4, (1, num * cp * Ndsym))
    # Xd1 = torch.zeros(4, num * cp * Ndsym).scatter_(dim=0, index=X_ori, value=1) + 0j
    # Xd1 = constellation @ Xd1
    # Xd1 = Xd1.view(num, cp, 1, Ndsym)

    Xp = torch.ones(num, cp, 1, 1) + 0j

    # X = torch.cat((Xp, Xd1), dim=3)

    X_ori = torch.randint(0, 4, (1, num * cd))
    Xd2 = torch.zeros(4, num * cd).scatter_(dim=0, index=X_ori, value=1) + 0j
    Xd2 = constellation @ Xd2
    Xd2 = Xd2.view(num, cd, 1, 1)
    X = torch.cat((Xp, Xd2), dim=1)
    Y = H @ X
    Y = Y + torch.sqrt(noise_var) * torch.randn_like(Y)
    # print(H_ener, noise_var)

    return Y, H, X, noise_var, H_ener


def main_naive(SNR_dB, chosen_cars, n, n_ant, alpha=1):
    num = 10
    cp = 4
    cd = 64 - cp
    batch_size = num
    Y, H, X, noise_var, phi = get_YHX_simple(SNR_dB, chosen_cars, num, n_ant, cp, cd, seed=1234)
    H1_label = H[:, :cp, :, :].reshape(num * cp, n_ant, 1)
    H2_label = H[:, cp:, :, :].reshape(num * cd, n_ant, 1)
    X1_label = X[:, :cp, :, :].reshape(num * cp, 1, 1)
    X2_label = X[:, cp:, :, :].reshape(num * cd, 1, 1)
    Y1 = Y[:, :cp, :, :].reshape(num * cp, n_ant, 1)
    Y2 = Y[:, cp:, :, :].reshape(num * cd, n_ant, 1)

    def mapping_1(x):
        x = x.reshape(batch_size, 4).mean(dim=1).reshape(batch_size, 1, 1).repeat(60, 1, 1)
        return x * alpha

    def mapping_2(x):
        x = x.reshape(batch_size, 60).mean(dim=1).reshape(batch_size, 1, 1).repeat(4, 1, 1)
        return x * alpha

    # model = Unfolding_NN(batch_size, cp, 64, 1, n_ant, 1, 1, 5, 10, 0,
    #                     1, 1, 1, 1, 1, None, None, None, 1,
    #                     writer=None, if_print=True, init_input_var=init_var,
    #                     var_mapping1=None, var_mapping2=None)
    model = Unfolding_NN(batch_size, cp, 64, 1, n_ant, 1, 1, 5, 10, noise_var,
                         1, 1, 1, 1, 1, None, None, None, 1,
                         writer=None, if_print=True, init_input_var=None,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)
    # var_mapping1=lambda x: 0.2 * x, var_mapping2=lambda x: 0.2 * x)
    H1_est = torch.zeros_like(H1_label)
    X1_est = X1_label
    var_H1 = phi * torch.ones(num * cp, 1, 1)
    # var_H1 = None
    var_X1 = torch.zeros(num * cp, 1, 1)
    X2_est = torch.ones_like(X2_label)
    var_X2 = torch.ones(num * cp, 1, 1)
    model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                  H2_label, Y1, Y2, noise_var)


def main_VI(SNR_dB, chosen_cars, n, n_ant, n_samples):
    num = 20
    cp = 4
    cd = 64 - cp
    Y, H, X, noise_var, phi = get_YHX_simple(SNR_dB, chosen_cars, num, n_ant, cp, cd, seed=1234)
    for i in range(num):
        H1_label = H[i, :cp, :, :].reshape(cp, n_ant, 1)
        H2_label = H[i, cp:, :, :].reshape(cd, n_ant, 1)
        X1_label = X[i, :cp, :, :].reshape(cp, 1, 1)
        X2_label = X[i, cp:, :, :].reshape(cd, 1, 1)
        Y1 = Y[i, :cp, :, :].reshape(cp, n_ant, 1)
        Y2 = Y[i, cp:, :, :].reshape(cd, n_ant, 1)
        model = Bidirection_VI(cp, 64, 1, n, 1, 1, 5, 10, noise_var, 1, 1, 1,
                               1, 1, None, None, None, 1, n_samples, writer=None, if_print=True)
        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        var_H1 = None
        var_X1 = torch.zeros(cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(cp, 1, 1)
        model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                      H2_label, Y1, Y2, noise_var)


# H_batch: num * all_ant * cp * 2
def get_deepunfolding_batch(H_batch, SNR_dB, mask, batch_size, cp, cd, n_ant):
    eners = torch.mean(H_batch * H_batch, dim=[1, 2, 3]).view(-1, 1, 1, 1) * 2
    H = H_batch[:, :n_ant, :, 0] + 1j * H_batch[:, :n_ant, :, 1]

    H = H @ (mask + 0j)
    H = rearrange(H, '(n ue) ant car -> n car ant ue', n=batch_size)

    noise_var = math.pow(10, -SNR_dB / 10) * eners

    constellation = torch.tensor([-1, 1]).view(1, -1) + 1j * torch.tensor([-1, 1]).view(-1, 1)
    constellation = constellation.view(1, 1, 1, -1) * 0.7071178

    Xp = torch.ones(batch_size, cp, 1, 1) + 0j

    X_ori = torch.randint(0, 4, (1, batch_size * cd))
    Xd2 = torch.zeros(4, batch_size * cd).scatter_(dim=0, index=X_ori, value=1) + 0j
    Xd2 = constellation @ Xd2
    Xd2 = Xd2.view(batch_size, cd, 1, 1)
    X = torch.cat((Xp, Xd2), dim=1)
    Y = H @ X
    Y = Y + torch.sqrt(noise_var) * torch.randn_like(Y)

    return Y, H, X, noise_var


def get_deepunfolding_dataloader(dataset, batch_size, n_ue=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size*n_ue, shuffle=True,
                                       num_workers=4, pin_memory=True, drop_last=True)


def train_deep_unfolding_epoch(model, dataloader, optimizer, SNR_dB, mask, batch_size, cp, cd, n_ant, phi):
    model.train()
    sum_results = torch.zeros(216)
    for i, H_batch in enumerate(dataloader):
        optimizer.zero_grad()
        Y, H, X, noise_var = get_deepunfolding_batch(H_batch[0], SNR_dB, mask, batch_size, cp, cd, n_ant)
        H1_label = H[:, :cp, :, :].reshape(batch_size * cp, n_ant, 1)
        H2_label = H[:, cp:, :, :].reshape(batch_size * cd, n_ant, 1)
        X1_label = X[:, :cp, :, :].reshape(batch_size * cp, 1, 1)
        X2_label = X[:, cp:, :, :].reshape(batch_size * cd, 1, 1)
        Y1 = Y[:, :cp, :, :].reshape(batch_size * cp, n_ant, 1)
        Y2 = Y[:, cp:, :, :].reshape(batch_size * cd, n_ant, 1)

        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        var_H1 = phi * torch.ones(batch_size * cp, 1, 1)
        var_X1 = torch.zeros(batch_size * cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(batch_size * cp, 1, 1)
        loss, H1_mse, H2_mse, X2_mse, X2_ber, write_message = \
            model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label,
                          H1_label, H2_label, Y1, Y2, noise_var)
        loss.backward()
        optimizer.step()
        model.input_var.data.clamp_(0.01,10)
        model.damping_params1.data.clamp_(0.01,0.99)
        model.damping_params2.data.clamp_(0.01,0.99)
        if torch.isnan(model.damping_params2).any():
            print(model.input_var, model.damping_params1, model.damping_params2)
            print(model.input_var.grad, model.damping_params1.grad, model.damping_params2.grad)
        with torch.no_grad():
            sum_results += torch.tensor(write_message)
    return sum_results / (i+1)


def test_deep_unfolding_epoch(model, dataloader, SNR_dB, mask, batch_size, cp, cd, n_ant, phi):
    model.eval()
    sum_results = torch.zeros(216)
    for i, H_batch in enumerate(dataloader):
        Y, H, X, noise_var = get_deepunfolding_batch(H_batch[0], SNR_dB, mask, batch_size, cp, cd, n_ant)
        H1_label = H[:, :cp, :, :].reshape(batch_size * cp, n_ant, 1)
        H2_label = H[:, cp:, :, :].reshape(batch_size * cd, n_ant, 1)
        X1_label = X[:, :cp, :, :].reshape(batch_size * cp, 1, 1)
        X2_label = X[:, cp:, :, :].reshape(batch_size * cd, 1, 1)
        Y1 = Y[:, :cp, :, :].reshape(batch_size * cp, n_ant, 1)
        Y2 = Y[:, cp:, :, :].reshape(batch_size * cd, n_ant, 1)

        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        # var_H1 = phi * torch.ones(batch_size * cp, 1, 1)
        var_H1 = None
        var_X1 = torch.zeros(batch_size * cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(batch_size * cp, 1, 1)
        loss, H1_mse, H2_mse, X2_mse, X2_ber, write_message = \
            model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label,
                          H1_label, H2_label, Y1, Y2, noise_var)
        with torch.no_grad():
            sum_results += torch.tensor(write_message)
    return sum_results / (i+1)


def train_unfolding(n_epoch, model, optimizer, datasets, SNR_dB, batch_size, chosen_cars,
                    cp, cd, n_ant, phi, writer, n_ue=1):
    mask = torch.zeros(1, cp + cd, cp + cd)
    j = cp
    k = 0
    for i in range(cp + cd):
        if k < cp and i == chosen_cars[k]:
            mask[0][i][k] = 1
            k += 1
        else:
            mask[0][i][j] = 1
            j += 1
    train_loader = get_deepunfolding_dataloader(datasets[0], batch_size, n_ue)
    test_loader = get_deepunfolding_dataloader(datasets[1], batch_size, n_ue)
    for e in range(n_epoch):
        results = \
            train_deep_unfolding_epoch(model, train_loader, optimizer, SNR_dB,
                                       mask, batch_size, cp, cd, n_ant, phi)
        results_test = \
            test_deep_unfolding_epoch(model, test_loader, SNR_dB,
                                      mask, batch_size, cp, cd, n_ant, phi)
        #print('Epoch %d: training H1 MSE %f, H2 MSE %f, X2 MSE %f, X2 BER %f'% (e, results[198], results[202], results[210], results[214]))
        #print('Testing H1 MSE %f, H2 MSE %f, X2 MSE %f, X2 BER %f'% (e, results_test[198], results_test[202], results_test[210], results_test[214]))
        writer.writerow(results)
        writer.writerow(results_test)
    return None


def main_unfolding(n_epoch, datasets, SNR_dB, batch_size, chosen_cars, NN_depth, unfolding_depth,
                   init_input_var, cp, cd, n_ant, phi, writer, lr):

    def mapping_1(x, alpha):
        x = x.reshape(batch_size, 4).mean(dim=1).reshape(batch_size, 1, 1).repeat(60, 1, 1)
        return x * alpha

    def mapping_2(x, alpha):
        x = x.reshape(batch_size, 60).mean(dim=1).reshape(batch_size, 1, 1).repeat(4, 1, 1)
        return x * alpha

    model = Unfolding_NN(batch_size, cp, cp+cd, 1, n_ant, 1, 1, NN_depth, unfolding_depth, 0,
                         1, 1, 1, 1, 1, None, None, None, 1,
                         writer=None, if_print=False, init_input_var=init_input_var,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)
    optimizer = torch.optim.Adam(model.unfolding_params, lr=lr)
    train_unfolding(n_epoch, model, optimizer, datasets, SNR_dB, batch_size, chosen_cars,
                    cp, cd, n_ant, phi, writer=writer, n_ue=1)
    torch.save(model.state_dict(), 'model_unfolding_0dB_gamma0.5.pth')
    return None


def main_test(datasets, SNR_dB, batch_size, chosen_cars, NN_depth, unfolding_depth,
              cp, cd, n_ant, phi, alpha, writer):
    def mapping_1(x):
        x = x.reshape(batch_size, 4).mean(dim=1).reshape(batch_size, 1, 1).repeat(60, 1, 1)
        return x * alpha

    def mapping_2(x):
        x = x.reshape(batch_size, 60).mean(dim=1).reshape(batch_size, 1, 1).repeat(4, 1, 1)
        return x * alpha
    model = Unfolding_NN(batch_size, cp, cp+cd, 1, n_ant, 1, 1, NN_depth, unfolding_depth, 0,
                         1, 1, 1, 1, 1, None, None, None, 1,
                         writer=writer, if_print=True, init_input_var=None,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)

    mask = torch.zeros(1, cp + cd, cp + cd)
    j = cp
    k = 0
    for i in range(cp + cd):
        if k < cp and i == chosen_cars[k]:
            mask[0][i][k] = 1
            k += 1
        else:
            mask[0][i][j] = 1
            j += 1
    test_loader = get_deepunfolding_dataloader(datasets[1], batch_size, 1)
    test_deep_unfolding_epoch(model, test_loader, SNR_dB,
                              mask, batch_size, cp, cd, n_ant, phi)


def test_VI(SNR_dB, chosen_cars, n_ant, n_samples, writer, ii):
    num = 100
    cp = 4
    cd = 64 - cp
    Y, H, X, noise_var, phi = get_YHX_simple(SNR_dB, chosen_cars, num, n_ant, cp, cd, seed=1234, start=ii*num)
    print(1111)
    for i in range(num):
        H1_label = H[i, :cp, :, :].reshape(cp, n_ant, 1)
        H2_label = H[i, cp:, :, :].reshape(cd, n_ant, 1)
        X1_label = X[i, :cp, :, :].reshape(cp, 1, 1)
        X2_label = X[i, cp:, :, :].reshape(cd, 1, 1)
        Y1 = Y[i, :cp, :, :].reshape(cp, n_ant, 1)
        Y2 = Y[i, cp:, :, :].reshape(cd, n_ant, 1)
        model = Bidirection_VI(cp, cp+cd, 1, n_ant, 1, 1, 5, 10, noise_var[i,0,0,0], 1, 1, 1,
                               1, 1, None, None, None, 1, n_samples, writer=writer, if_print=False)
        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        var_H1 = None
        var_X1 = torch.zeros(cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(cp, 1, 1)
        model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                      H2_label, Y1, Y2, noise_var[i,0,0,0])
        if i % 100 == 0:
            print(i)


torch.set_num_threads(8)
n_car = 4
n_ant = 8
in_car = []
for i in range(0, 64, 63//(n_car-1)):
    in_car.append(i)
# main_VI(2, in_car, n_car, n_ant, 128)
phi = 8.88
init_var = [0.5] * 19
# var = 2.
# for i in range(10):
#     init_var.append(var)
#     init_var.append(var)
#     var *= 0.7
datasets = get_dataset(ratio=[0.8, 0.2], seed=1234, dataset_path='/mnt/HD2/yyz/MIMOnoisedata/',
                       name='32ant_64car_300k')
#SNR_list = [2, 0, -2, -4, -6, -8, -10]
SNR_list = [-4]
alphas = [0.2]
# alphas = [0.05, 0.1, 0.2, 0.4, 1, 2]
# name_prefix = 'result/VI_8ant_'
'''
name_prefix = 'result/testtttt_4ant_'
for SNR in SNR_list:
    name = name_prefix + 'SNR%d.csv' % SNR
    f = open(name, 'w')
    w = csv.writer(f)
    # for ii in range(50):
    #     test_VI(SNR_dB=SNR, chosen_cars=in_car, n_ant=8, n_samples=128, writer=w, ii=ii)
    for alpha in alphas:
        # name = name_prefix + 'SNR%d_idealSE.csv' % SNR
        name = name_prefix + 'SNR%d_alpha%.2f.csv' % (SNR, alpha)
        f = open(name, 'w')
        w = csv.writer(f)
        main_test(datasets=datasets, SNR_dB=SNR, batch_size=128, chosen_cars=in_car,
                  NN_depth=5, unfolding_depth=10, cp=n_car, cd=64-n_car,
                  n_ant=n_ant, phi=phi, alpha=alpha, writer=w)
exit(1)
main_naive(2, in_car, n_car, n_ant)
exit(1)
'''
f = open('unfolding_result_0dB_gamma0.5.csv', 'w')
w = csv.writer(f)
main_unfolding(n_epoch=50, datasets=datasets, SNR_dB=0, batch_size=512, chosen_cars=in_car,
               NN_depth=5, unfolding_depth=10, init_input_var=init_var, cp=n_car, cd=64-n_car,
               n_ant=n_ant, phi=phi, writer=w, lr=0.001)
