import torch
import torch.nn as nn
import math
import einops


from utils import MSE, BER


class BiGaBP_unfolding_iter(nn.Module):
    def __init__(self, cp, c, kp, k, alpha_init, beta_init, gamma_init, eta_init, phi, demod,
                 demod_H, bs=1, first_data_layer=False):
        super().__init__()
        # m: Nue, n: Nant
        # Xp: c x m x kp
        # H_pri: c x n x m
        # Y: c x n x k
        self.cp = cp
        self.c = c
        self.kp = kp
        self.k = k
        self.phi = phi  # priori average channel gain

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.beta = nn.Parameter(torch.ones(1) * beta_init)
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init)
        self.eta = nn.Parameter(torch.ones(1) * eta_init)  # Maybe used
        self.demod = demod
        self.demod_H = demod_H
        self.pilot_mask = torch.ones(self.c, 1, 1, self.k)
        self.pilot_mask[:cp, :, :, :kp] = 0
        self.pilot_mask = self.pilot_mask.repeat(bs, 1, 1, 1)
        self.pilot_mask_inv = 1 - self.pilot_mask
        self.first_data_layer = first_data_layer

    def FN(self, H_est, X_est, var_X, var_H, Y, N0):
        err, xi_y = self.Y_est(H_est, X_est, var_X, var_H, Y, N0)
        xi_x = xi_y + var_H
        xi_h = xi_y + self.phi * var_X
        # print('err', MSE(err), torch.mean(xi_y))
        return err, xi_x, xi_h

    def Y_est(self, H_est, X_est, var_X, var_H, Y, N0):
        HX = H_est * X_est
        HX_ = torch.sum(HX, dim=2, keepdim=True)
        err = Y.unsqueeze(2) - HX_ + HX
        tmp = H_est * H_est.conj() * var_X + \
              var_H * (X_est * X_est.conj() + var_X)
        tmp = tmp.real
        xi_y = torch.sum(tmp, dim=2, keepdim=True) - tmp + N0
        return err, xi_y

    def VN_X(self, H_est, X_est, var_X, err, xi_x):
        tmp = H_est.conj() / xi_x
        var_tmp = tmp * H_est
        var = torch.sum(var_tmp, dim=1, keepdim=True) - var_tmp
        var = 1 / var
        tmp_est = tmp * err
        est = torch.sum(tmp_est, dim=1, keepdim=True) - tmp_est
        est = est * var
        # est_post, var_post = self.demod(est, var)
        est_post, var_post = self.demod(est, self.gamma)
        X_est = X_est + self.eta * ((est_post - X_est) * self.pilot_mask)
        var_X = var_X + self.eta * ((var_post - var_X) * self.pilot_mask)
        return X_est, var_X

    def VN_H(self, H_est, X_est, var_H, err, xi_h):
        mask = self.alpha * self.pilot_mask_inv + self.beta * self.pilot_mask
        tmp = mask * X_est.conj() / xi_h
        var_tmp = tmp * X_est
        var = torch.sum(var_tmp, dim=3, keepdim=True) - var_tmp
        var = 1 / var
        tmp_est = tmp * err
        est = torch.sum(tmp_est, dim=3, keepdim=True) - tmp_est
        est = est * var
        est_post, var_post = self.demod_H(est, var, self.phi)
        H_est = H_est + (est_post - H_est) * self.eta
        var_H = var_H + (var_post - var_H) * self.eta
        return H_est, var_H

    def forward(self, H_est, X_est, var_X, var_H, Y, N0):
        # est = X_est.mean(dim=1)
        err, xi_x, xi_h = self.FN(H_est, X_est, var_X, var_H, Y, N0)
        X_est_new, var_X_new = self.VN_X(H_est, X_est, var_X, err, xi_x)
        # est = X_est_new.mean(dim=1)
        if self.first_data_layer:
            return H_est, X_est_new, var_X_new, var_H
        else:
            H_est_new, var_H_new = self.VN_H(H_est, X_est, var_H, err, xi_h)
            return H_est_new, X_est_new, var_X_new, var_H_new.real


class BiGaBP_unfolding_lastiter(nn.Module):
    def __init__(self, cp, c, kp, k, alpha_init, beta_init, gamma_init, phi, demod, demod_H, bs=1):
        super().__init__()
        # m: Nue, n: Nant
        # Xp: c x m x kp
        # H_pri: c x n x m
        # Y: c x n x k
        self.cp = cp
        self.c = c
        self.kp = kp
        self.k = k
        self.phi = phi  # priori average channel gain

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.beta = nn.Parameter(torch.ones(1) * beta_init)
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init)
        self.demod = demod
        self.demod_H = demod_H
        self.pilot_mask = torch.ones(self.c, 1, 1, self.k)
        self.pilot_mask[:cp, :, :, :kp] = 0
        self.pilot_mask = self.pilot_mask.repeat(bs, 1, 1, 1)
        self.pilot_mask_inv = 1 - self.pilot_mask

    def FN(self, H_est, X_est, var_X, var_H, Y, N0):
        err, xi_y = self.Y_est(H_est, X_est, var_X, var_H, Y, N0)
        xi_x = xi_y + var_H
        xi_h = xi_y + self.phi * var_X
        return err, xi_x, xi_h

    def Y_est(self, H_est, X_est, var_X, var_H, Y, N0):
        HX = H_est * X_est
        HX_ = torch.sum(HX, dim=2, keepdim=True)
        err = Y.unsqueeze(2) - HX_ + HX
        tmp = H_est * H_est.conj() * var_X + \
              var_H * (X_est * X_est.conj() + var_X)
        xi_y = torch.sum(tmp, dim=2, keepdim=True) - tmp + N0
        return err, xi_y

    def forward(self, H_est, X_est, var_X, var_H, Y, N0):
        mask = self.alpha * self.pilot_mask_inv + self.beta * self.pilot_mask

        err, xi_x, xi_h = self.FN(H_est, X_est, var_X, var_H, Y, N0)

        tmp = H_est.conj() / xi_x
        var_tmp = tmp * H_est
        var = 1 / torch.sum(var_tmp, dim=1)
        est = torch.sum(tmp * err, dim=1) * var
        est_post, var_post = self.demod(est, self.gamma)
        X_est_ = est_post * self.pilot_mask[:, 0, :, :]
        var_X_ = var_post * self.pilot_mask[:, 0, :, :]

        tmp = mask * X_est.conj() / xi_h
        var_tmp = tmp * X_est
        var = 1 / torch.sum(var_tmp, dim=3)
        est = torch.sum(tmp * err, dim=3) * var
        H_est_, var_H_ = self.demod_H(est, var, self.phi)

        return H_est_, X_est_, var_X_, var_H_.real  # pilot masked


class BiGaBP_unfolding(nn.Module):
    def __init__(self, cp, c, m, n, kp, k, alpha_init, beta_init, gamma_init,
                 eta_init, phi, demod, demod_hard, demod_H, n_layer, bs=1,
                 writer=None, if_print=False, isest1=True, init_input_var=None,
                 first_data_layer=False):
        super().__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([])
        for i in range(n_layer - 1):
            self.layers.append(BiGaBP_unfolding_iter(cp, c, kp, k, alpha_init[i],
                                                     beta_init[i], gamma_init[i], eta_init[i],
                                                     phi, demod, demod_H, bs,
                                                     first_data_layer=(first_data_layer & (i == 0))))
        self.last_layer = BiGaBP_unfolding_lastiter(cp, c, kp, k, alpha_init[n_layer-1],
                                                    beta_init[n_layer-1], gamma_init[n_layer-1], phi, demod_hard, demod_H, bs)
        self.m = m
        self.n = n
        self.k = k
        self.writer = writer
        self.if_print = if_print
        self.if_write = (writer is not None)
        self.ratio = 1 - (cp * kp / (c * k))
        self.n_layer = n_layer
        self.isest1 = isest1
        if init_input_var is None:
            self.require_var = False
        else:
            self.require_var = True
            self.input_var = nn.Parameter(torch.ones(1) * init_input_var)

    def print(self, H_est, X_est, i, X_label, H_label):
        pred_X = torch.mean(X_est, dim=1)
        MSE_X = MSE(pred_X - X_label) / self.ratio
        BER_X = BER(X_label, pred_X) / self.ratio
        MSE_H = MSE(torch.mean(H_est, dim=3) - H_label)

        if self.if_write:
            if self.isest1:
                self.writer.writerow([i, MSE_X, BER_X, MSE_H, 0, 0, 0, 'est1'])
            else:
                self.writer.writerow([i, 0, 0, 0, MSE_X, BER_X, MSE_H, 'est2'])
        if self.if_print:
            print('Iteration', i, 'finished')
            print('MSE of data: ', MSE_X)
            print('BER of data: ', BER_X)
            print('MSE of channel matrix: ', MSE_H)

    def forward(self, H_est, X_est, var_X, var_H, Y, N0, X_label, H_label):
        N0 = N0.unsqueeze(2)
        if self.require_var:
            var_H = self.input_var * torch.ones_like(H_est)
        X_est = X_est.unsqueeze(1).repeat([1, self.n, 1, 1])
        var_X = var_X.unsqueeze(1).repeat([1, self.n, 1, 1])
        H_est = H_est.unsqueeze(3).repeat([1, 1, 1, self.k])
        var_H = var_H.unsqueeze(3).repeat([1, 1, 1, self.k])
        for (i, l) in enumerate(self.layers):
            H_est, X_est, var_X, var_H = l.forward(H_est, X_est, var_X, var_H, Y, N0)
            #print(X_est[0,0,:,5:])
            # print(X_label[0,:,5:])
            # print(H_est.shape, X_est.shape, var_X.shape, var_H.shape)
            # print(var_H.mean(), var_X.mean())
            # print(MSE(H_est.mean(dim=3) @ X_est.mean(dim=1)-Y))
            self.print(H_est, X_est, i, X_label, H_label)
        H_est, X_est, var_X, var_H = self.last_layer.forward(H_est, X_est, var_X, var_H, Y, N0)
        # print(H_est.shape, X_est.shape, var_X.shape, var_H.shape)
        # exit(1)
        return H_est, X_est, var_X, var_H
