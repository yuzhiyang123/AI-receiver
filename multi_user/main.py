import torch
from einops import rearrange
import math
import numpy as np
import csv

from semi_blind import *
from utils import *


def get_alpha_beta_gamma_phi(unfolding_depth, n_layer, T,
                             tau0, tau1, tau2, eta_0, kp, kd):
    alpha = []
    beta = []
    gamma = []
    eta = []
    for i in range(unfolding_depth):
        alpha.append([])
        beta.append([])
        gamma.append([])
        eta.append([eta_0] * (n_layer - 1))
        for j in range(n_layer):
            t = i * n_layer + j
            if t < T:
                b = tau2 + (1 - tau2) * t / T
                a = 1 + (1 - b) * kd / kp
                c = tau0 + tau1 * t / T
            else:
                a = 1
                b = 1
                c = tau0 + tau1
            alpha[i*2].append(a)
            beta[i*2].append(b)
            gamma[i*2].append(c)
        alpha.append([])
        beta.append([])
        gamma.append([])
        eta.append([eta_0] * (n_layer - 1))
        for j in range(n_layer):
            t = i * n_layer + j
            if t < T:
                b = tau2 + (1 - tau2) * t / T
                a = 1 + (1 - b) * kd / kp
                c = tau0 + tau1 * t / T
            else:
                a = 1
                b = 1
                c = tau0 + tau1
            alpha[i*2+1].append(a)
            beta[i*2+1].append(b)
            gamma[i*2+1].append(c)
    return alpha, beta, gamma, eta


def get_pilot(length, n):
    phi = torch.arange(length).view(1, -1) * torch.arange(n).view(-1, 1)
    phi = (2 * 3.1415926 / length) * phi
    return torch.exp(1j * phi)


def get_channels(num, n_ant, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='32ant_64car_300k',start=0):
    dest_test = "channel_test_" + name + ".pt"
    channel_test = torch.load(dest_test)
    ener = torch.mean(channel_test*channel_test)
    total_data = channel_test.shape[0]
    torch.manual_seed(seed)
    perm = torch.randperm(total_data)
    ids = perm[start:start+num]
    return channel_test[ids, :n_ant, :, :], ener


def get_YHX(SNR_dB, chosen_cars, num, n_ant, cp, cd, Nue, Ndsym, Npsym,
            seed=1234, dataset_path='/mnt/HD2/yyz/MIMOsemiblinddata/', start=0):
    # H = torch.load("channels.pth")
    H, H_ener = get_channels(num*Nue, n_ant, seed, dataset_path, start=start)

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
    eners = torch.mean(H.real * H.real + H.imag * H.imag, dim=[1,2,3]).view(-1, 1, 1, 1)

    noise_var = math.pow(10, -SNR_dB / 10) * Nue * eners
    # print(noise_var)

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
    Y = Y + torch.sqrt(noise_var) * torch.randn_like(Y)
    # print(H_ener, noise_var)

    return Y, H, X, noise_var, H_ener


def get_YHX_simple(SNR_dB, chosen_cars, num, n_ant, cp, cd, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOsemiblinddata/'):
    H, H_ener = get_channels(num, n_ant, seed=seed)

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

    Xp = torch.ones(num, cp, 1, 1) + 0j

    X_ori = torch.randint(0, 4, (1, num * cd))
    Xd2 = torch.zeros(4, num * cd).scatter_(dim=0, index=X_ori, value=1) + 0j
    Xd2 = constellation @ Xd2
    Xd2 = Xd2.view(num, cd, 1, 1)
    X = torch.cat((Xp, Xd2), dim=1)
    Y = H @ X
    Y = Y + torch.sqrt(noise_var) * torch.randn_like(Y)
    # print(H_ener, noise_var)

    return Y, H, X, noise_var, H_ener


def demod_H(est, var, phi):
    tmp = phi / (phi + var)
    return est * tmp, var * tmp


def main_naive_simple(SNR_dB, chosen_cars, n_ant):
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
        return x * 1

    def mapping_2(x):
        x = x.reshape(batch_size, 60).mean(dim=1).reshape(batch_size, 1, 1).repeat(4, 1, 1)
        return x * 1
    model = Unfolding_NN(batch_size, cp, 64, 1, n_ant, 1, 1, 5, 10,
                         1, 1, 1, 1, 1, None, None, None, 1,
                         writer=None, if_print=True, init_input_var=None,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)
    H1_est = torch.zeros_like(H1_label)
    X1_est = X1_label
    var_H1 = phi * torch.ones(num * cp, 1, 1)
    # var_H1 = None
    var_X1 = torch.zeros(num * cp, 1, 1)
    X2_est = torch.ones_like(X2_label)
    var_X2 = torch.ones(num * cp, 1, 1)
    model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                  H2_label, Y1, Y2, noise_var)


def main_VI_simple(SNR_dB, chosen_cars, n_ant, n_samples):
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
        model = Bidirection_VI(cp, 64, 1, n_ant, 1, 1, 5, 10, 1, 1, 1,
                               1, 1, None, None, None, 1, n_samples,
                               writer=None, if_print=True)
        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        var_H1 = None
        var_X1 = torch.zeros(cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(cp, 1, 1)
        model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                      H2_label, Y1, Y2, noise_var)


def main_naive(SNR_dB, chosen_cars, n_ant, cp, c, n_ue, kp, k,
               unfolding_depth, n_layer, T,
               tau0, tau1, tau2, eta_0):
    num = 1
    cd = c - cp
    batch_size = num
    Y, H, X, noise_var, phi = get_YHX(SNR_dB, chosen_cars, num, n_ant, cp, cd, n_ue, k-kp, kp, seed=1234)
    # print(Y.shape, H.shape, X.shape, noise_var.shape)
    H1_label = H[:, :cp, :, :].reshape(num * cp, n_ant, n_ue)
    H2_label = H[:, cp:, :, :].reshape(num * cd, n_ant, n_ue)
    X1_label = X[:, :cp, :, :].reshape(num * cp, n_ue, k)
    X2_label = X[:, cp:, :, :].reshape(num * cd, n_ue, k)
    Y1 = Y[:, :cp, :, :].reshape(num * cp, n_ant, k)
    Y2 = Y[:, cp:, :, :].reshape(num * cd, n_ant, k)

    alpha, beta, gamma, eta = get_alpha_beta_gamma_phi(unfolding_depth, n_layer, T,
                                                        tau0, tau1, tau2, eta_0, kp, k-kp)

    def mapping_1(x):
        x = x.reshape(batch_size, cp * n_ant * n_ue).mean(dim=1).reshape(batch_size, 1, 1).repeat(cd, n_ant, n_ue)
        return x * 1

    def mapping_2(x):
        x = x.reshape(batch_size, cd * n_ant * n_ue).mean(dim=1).reshape(batch_size, 1, 1).repeat(cp, n_ant, n_ue)
        return x * 1

    model = Unfolding_NN(batch_size, cp, c, n_ue, n_ant, kp, k,
                         5, unfolding_depth,
                         alpha, beta, gamma, eta, phi,
                         demod_QPSK2, demod_QPSK2, demod_H, n_layer,
                         writer=None, if_print=True, init_input_var=None,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)
    Np = noise_var.repeat([1, cp, 1, 1]).reshape(-1, 1, 1)
    H1_est, var_H1 = LS(X1_label[:, :, :kp], Y1[:, :, :kp], Np, kp, Es=1)
    # H1_est = torch.zeros_like(H1_label)
    X1_est = torch.zeros_like(X1_label)
    # var_H1 = phi * torch.ones(num * cp, n_ant, n_ue)
    # var_H1 = None
    var_X1 = torch.ones(num * cp, n_ue, k)
    X1_est[:, :, :kp] = X1_label[:, :, :kp]
    var_X1[:, :, :kp] = 0
    X2_est = torch.zeros_like(X2_label)
    var_X2 = torch.ones(num * cd, n_ue, k)
    model.forward(H1_est, X1_est, var_X1, var_H1, X2_est,
                  var_X2, X1_label, X2_label, H1_label,
                  H2_label, Y1, Y2, noise_var)


def main_VI(SNR_dB, chosen_cars, n_ant, n_samples, cp, c, n_ue, kp, k,
               unfolding_depth, n_layer, T,
               tau0, tau1, tau2, eta_0):
    num = 20
    cd = c - cp
    Y, H, X, noise_var, phi = get_YHX(SNR_dB, chosen_cars, num, n_ant, cp, cd, n_ue, k-kp, kp, seed=1234)
    alpha, beta, gamma, eta = get_alpha_beta_gamma_phi(unfolding_depth, n_layer, T,
                                                        tau0, tau1, tau2, eta_0, kp, k-kp)
    for i in range(num):
        H1_label = H[i, :cp, :, :].reshape(cp, n_ant, 1)
        H2_label = H[i, cp:, :, :].reshape(cd, n_ant, 1)
        X1_label = X[i, :cp, :, :].reshape(cp, 1, 1)
        X2_label = X[i, cp:, :, :].reshape(cd, 1, 1)
        Y1 = Y[i, :cp, :, :].reshape(cp, n_ant, 1)
        Y2 = Y[i, cp:, :, :].reshape(cd, n_ant, 1)
        model = Bidirection_VI(cp, c, n_ue, n_ant, kp, k,
                               5, unfolding_depth, alpha, beta, gamma, eta, phi,
                               demod_QPSK, demod_QPSK, demod_H, n_layer, n_samples,
                               writer=None, if_print=True)
        H1_est = torch.zeros_like(H1_label)
        X1_est = X1_label
        var_H1 = None
        var_X1 = torch.zeros(cp, 1, 1)
        X2_est = torch.ones_like(X2_label)
        var_X2 = torch.ones(cp, 1, 1)
        model.forward(H1_est, X1_est, var_X1, var_H1, X2_est, var_X2, X1_label, X2_label, H1_label,
                      H2_label, Y1, Y2, noise_var)


def main_test(SNR_dB, chosen_cars, n_ant, cp, c, n_ue, kp, k,
              unfolding_depth, n_layer, T,
              tau0, tau1, tau2, eta_0, writer=None):
    num = 6
    cd = c - cp
    batch_size = num

    alpha, beta, gamma, eta = get_alpha_beta_gamma_phi(unfolding_depth, n_layer, T,
                                                        tau0, tau1, tau2, eta_0, kp, k-kp)

    def mapping_1(x):
        x = x.reshape(batch_size, cp * n_ant * n_ue).mean(dim=1).reshape(batch_size, 1, 1).repeat(cd, n_ant, n_ue)
        return x * 1

    def mapping_2(x):
        x = x.reshape(batch_size, cd * n_ant * n_ue).mean(dim=1).reshape(batch_size, 1, 1).repeat(cp, n_ant, n_ue)
        return x * 1

    model = Unfolding_NN(batch_size, cp, c, n_ue, n_ant, kp, k,
                         5, unfolding_depth,
                         alpha, beta, gamma, eta, 8.8,
                         demod_QPSK2, demod_QPSK2, demod_H, n_layer,
                         writer=writer, if_print=False, init_input_var=None,
                         var_mapping1=mapping_1, var_mapping2=mapping_2)

    for i in range(21576 // (n_ue * num)):
        Y, H, X, noise_var, _ = get_YHX(SNR_dB, chosen_cars, num, n_ant, cp, cd, n_ue,
                                     k-kp, kp, seed=1234, start=i*n_ue)
        # print(Y.shape, H.shape, X.shape, noise_var.shape)
        H1_label = H[:, :cp, :, :].reshape(num * cp, n_ant, n_ue)
        H2_label = H[:, cp:, :, :].reshape(num * cd, n_ant, n_ue)
        X1_label = X[:, :cp, :, :].reshape(num * cp, n_ue, k)
        X2_label = X[:, cp:, :, :].reshape(num * cd, n_ue, k)
        Y1 = Y[:, :cp, :, :].reshape(num * cp, n_ant, k)
        Y2 = Y[:, cp:, :, :].reshape(num * cd, n_ant, k)

        Np = noise_var.repeat([1, cp, 1, 1]).reshape(-1, 1, 1)
        H1_est, var_H1 = LS(X1_label[:, :, :kp], Y1[:, :, :kp], Np, kp, Es=1)
        # H1_est = torch.zeros_like(H1_label)
        X1_est = torch.zeros_like(X1_label)
        # var_H1 = phi * torch.ones(num * cp, n_ant, n_ue)
        # var_H1 = None
        var_X1 = torch.ones(num * cp, n_ue, k)
        X1_est[:, :, :kp] = X1_label[:, :, :kp]
        var_X1[:, :, :kp] = 0
        X2_est = torch.zeros_like(X2_label)
        var_X2 = torch.ones(num * cd, n_ue, k)
        model.forward(H1_est, X1_est, var_X1, var_H1, X2_est,
                      var_X2, X1_label, X2_label, H1_label,
                      H2_label, Y1, Y2, noise_var)
        if i % 100 == 0:
            print(i)


#SNR_dB = 10
cp = 4
c = 64
n_ant = 8
n_ue = 4
kp = 6
k = 36
unfolding_depth = 2
n_layer = 15
T = 15
tau0 = 1
tau1 = 2
tau2 = 0.4
eta_0 = 0.5
n_samples = 128

in_car = []
for i in range(0, 64, 63//(cp-1)):
    in_car.append(i)
SNR_list = [20, 15, 10, 5]
alphas = [0.1]
# alphas = [0.05, 0.1, 0.2, 0.4, 1, 2]
name_prefix = 'result/result_naive2_'
# name_prefix = 'result/single_user_4ant_'
for SNR in SNR_list:
    name = name_prefix + 'SNR%d.csv' % SNR
    f = open(name, 'w')
    w = csv.writer(f)
    main_test(SNR, in_car, n_ant, cp, c, n_ue, kp, k,
              unfolding_depth, n_layer, T,
              tau0, tau1, tau2, eta_0, writer=w)
# main_naive(SNR_dB, in_car, n_ant, cp, c, n_ue, kp, k,
#            unfolding_depth, n_layer, T,
#            tau0, tau1, tau2, eta_0)
# main_VI(SNR_dB, in_car, n_ant, n_samples, cp, c, n_ue, kp, k,
#         unfolding_depth, n_layer, T,
#         tau0, tau1, tau2, eta_0)

# main_naive_simple(2, in_car, n_ant)
# # main_VI_simple(2, in_car, n_ant, 128)
