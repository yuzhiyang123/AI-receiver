import torch.nn as nn
import torch
from einops import rearrange

from basic_modules import BiGaBP_unfolding
from model.model import get_mixer_net
from utils import NMSE, init_mapping_net, demod_QPSK, LS


MSE = lambda H: torch.mean(H.real * H.real + H.imag * H.imag)
MSEre = lambda H: torch.mean(H * H).item()
MSE3D = lambda H: torch.mean(H.real * H.real + H.imag * H.imag, dim=[1, 2])
MSE4D = lambda H: torch.mean(H.real * H.real + H.imag * H.imag, dim=[1, 2, 3])


def BER(target, est):
    err_re = est.real.mul(target.real).ge(0)
    err_im = est.imag.mul(target.imag).ge(0)
    err = err_re * err_im
    return 1 - torch.mean(err.float()).item()


def BER3D(target, est):
    err_re = est.real.mul(target.real).ge(0)
    err_im = est.imag.mul(target.imag).ge(0)
    err = err_re * err_im
    return 1 - torch.mean(err.float(), dim=[1,2])


def merge_est(est1, var1, est2, var2):
    var = var1 * var2 / (var1 + var2)
    est = est1 / var1 + est2 / var2
    return est * var, var


class Est_data(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, H, X_est, var_X, H_var, Y, N0, X_label, H_label):
        # H: c x ant x 1
        # H_var: c x 1 x 1
        H_ener = torch.sum(H.real * H.real + H.imag * H.imag, dim=[1, 2], keepdim=True)
        X_est = (H.mH @ Y) / H_ener
        X_var = (H_var + N0) / H_ener
        X_est, X_var = demod_QPSK(X_est, X_var)
        H_newpred = Y * X_est.conj()
        H_newvar = H_ener * X_var + N0
        # H, H_var = merge_est(H, H_var, H_newpred, H_newvar)
        # return H, X_est, X_var, H_var
        return H_newpred, X_est, X_var, H_newvar


class Est_pilot(nn.Module):
    def __init__(self):
        # Xp: c x 1 x k
        # H: c x ant x 1
        # Y: c x ant x k
        super().__init__()

    def forward(self, H, X_est, var_X, H_var, Y, N0, Xp, H_label):
        # H: c x ant x 1
        # H_var: c x 1 x 1
        H_newpred = Y * Xp.conj()
        H_newvar = N0
        if H_var is not None:
            H, H_var = merge_est(H, H_var, H_newpred, H_newvar)
            return H, None, None, H_var
        else:
            return H_newpred, None, None, H_newvar


class Unfolding_NN(nn.Module):
    def __init__(self, batch_size, cp, c, m, n, kp, k, NN_depth, unfolding_depth,
                 alpha_init, beta_init, gamma_init,
                 eta_init, phi, demod, demod_hard, demod_H, n_layer,
                 writer=None, if_print=False, init_input_var=None,
                 var_mapping1=lambda x: x, var_mapping2=lambda x: x):
        super().__init__()
        self.batch_size = batch_size
        self.unfolding_depth = unfolding_depth
        self.var_mapping1 = var_mapping1
        self.var_mapping2 = var_mapping2
        self.if_print = if_print
        if init_input_var is None:
            self.is_unfolding = False
        else:
            self.is_unfolding = True
            self.damping_params1 = nn.Parameter(torch.ones(unfolding_depth-1) * 0.5)
            self.damping_params2 = nn.Parameter(torch.ones(unfolding_depth-1) * 0.5)
        if init_input_var is None:
            init_input_var = [None] * (unfolding_depth * 2)
        self.layers = nn.ModuleList([])
        cd = c - cp
        self.cp = cp
        self.cd = cd
        self.kd = k - kp
        self.phi = phi
        multiuser = (m > 1)
        # print(unfolding_depth)
        # print(len(alpha_init), len(beta_init),len(gamma_init),len(eta_init),len(init_input_var))
        for i in range(unfolding_depth-1):
            mod = get_unfolding_model(cp, cp, m, n, kp, k, alpha_init[i*2], beta_init[i*2], gamma_init[i*2],
                                      eta_init[i*2], phi, demod, demod_hard, demod_H, n_layer,
                                      bs=batch_size, writer=None, if_print=if_print, isest1=True,
                                      init_input_var=init_input_var[i*2], multiuser=multiuser)
            self.layers.append(mod)
            mod = get_mixer_net(n, cp, n, cd, NN_depth)
            init_mapping_net(mod, is_mapping1=True)
            self.layers.append(mod)
            mod = get_unfolding_model(0, cd, m, n, 0, k, alpha_init[i*2+1], beta_init[i*2+1], gamma_init[i*2+1],
                                      eta_init[i*2+1], phi, demod, demod_hard, demod_H, n_layer,
                                      bs=batch_size, writer=None, if_print=if_print, isest1=False,
                                      init_input_var=init_input_var[i*2+1], multiuser=multiuser,
                                      first_data_layer=(i == 0))
            self.layers.append(mod)
            mod = get_mixer_net(n, cd, n, cp, NN_depth)
            init_mapping_net(mod, is_mapping1=False)
            self.layers.append(mod)
        mod = get_unfolding_model(cp, cp, m, n, kp, k, alpha_init[unfolding_depth*2-2],
                                  beta_init[unfolding_depth*2-2], gamma_init[unfolding_depth*2-2],
                                  eta_init[unfolding_depth*2-2], phi, demod, demod_hard, demod_H, n_layer,
                                  bs=batch_size, writer=None, if_print=if_print, isest1=True,
                                  init_input_var=init_input_var[unfolding_depth*2-2], multiuser=multiuser)
        self.layers.append(mod)
        mod = get_mixer_net(n, cp, n, cd, NN_depth)
        init_mapping_net(mod, is_mapping1=True)
        self.layers.append(mod)
        mod = get_unfolding_model(0, cd, m, n, 0, k, alpha_init[unfolding_depth*2-1],
                                  beta_init[unfolding_depth*2-1], gamma_init[unfolding_depth*2-1],
                                  eta_init[unfolding_depth*2-1], phi, demod, demod_hard, demod_H, n_layer,
                                  bs=batch_size, writer=None, if_print=if_print, isest1=False,
                                  init_input_var=init_input_var[unfolding_depth*2-1], multiuser=multiuser)
        self.layers.append(mod)
        title = []
        title.extend(['H1_MSE', 'H1_MSE2', 'H1_VAR', 'H1_VAR2'])
        title.extend(['mapped H2_MSE', 'H2_MSE2', 'H2_VAR', 'H2_VAR2'])
        title.extend(['H2_MSE', 'H2_MSE2', 'H2_VAR', 'H2_VAR2'])
        title.extend(['X2_MSE', 'X2_MSE2', 'X2_VAR', 'X2_VAR2', 'X2_BER', 'X2_BER2'])
        for i in range(self.unfolding_depth - 1):
            title.extend(['mapped H1_MSE', 'H1_MSE2', 'H1_VAR', 'H1_VAR2'])
            title.extend(['H1_MSE', 'H1_MSE2', 'H1_VAR', 'H1_VAR2'])
            title.extend(['mapped H2_MSE', 'H2_MSE2', 'H2_VAR', 'H2_VAR2'])
            title.extend(['H2_MSE', 'H2_MSE2', 'H2_VAR', 'H2_VAR2'])
            title.extend(['X2_MSE', 'X2_MSE2', 'X2_VAR', 'X2_VAR2', 'X2_BER', 'X2_BER2'])
        if writer is not None:
            writer.writerow(title)
        self.writer = writer

    def complex2real(self, H):
        H = rearrange(H, '(bs car) ant (u n) -> (bs u) ant car n', bs=self.batch_size, n=1)
        H = torch.cat((H.real, H.imag), dim=3)
        return H

    def real2complex(self, H):
        H_complex = H[:, :, :, 0] + 1j * H[:, :, :, 1]
        H_complex = rearrange(H_complex, '(bs u) ant car -> (bs car) ant u', bs=self.batch_size)
        return H_complex

    def forward(self, H1_est, X1_est, var_X1, var_H1, X2_est, var_X2,
                X1_label, X2_label, H1_label, H2_label, Y1, Y2, N0):
        write_message = []
        Np = N0.repeat([1, self.cp, 1, 1]).reshape(-1, 1, 1)
        Nd = N0.repeat([1, self.cd, 1, 1]).reshape(-1, 1, 1)
        loss = 0
        H1_est, X1_est, var_X1, var_H1 = self.layers[0].forward(H1_est, X1_est, var_X1, var_H1, Y1,
                                                                Np, X1_label, H1_label)
        H1_mse = MSE3D(H1_label - H1_est)
        X1_mse = MSE3D(X1_label - X1_est)
        X1_ber = BER3D(X1_label, X1_est)
        if self.if_print:
            print('Iteration 1, H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
            print('X1 MSE', 'X1 var', 'X1 BER')
        write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
        write_message.extend([X1_mse.mean().item(), MSEre(X1_mse), var_X1.mean().item(), MSEre(var_X1),
                              X1_ber.mean().item(), MSEre(X1_ber)])
        # loss = loss + H1_mse / H1_mse.detach()

        H1 = self.complex2real(H1_est)
        H2 = self.layers[1].forward(H1)
        H2_est = self.real2complex(H2)
        if self.is_unfolding:
            var_H2 = None
        else:
            var_H2 = self.var_mapping1(var_H1)
        dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
        if self.if_print:
            print('Iteration 1, H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
        write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])

        H2_est, X2_est, var_X2, var_H2 = self.layers[2].forward(H2_est, X2_est, var_X2, dist,
                                                                Y2, Nd, X2_label, H2_label)
        # H2_est, X2_est, var_X2, var_H2 = self.layers[2].forward(H2_est, X2_est, var_X2, var_H2,
        #                                                         Y2, X2_label, H2_label)
        mse = MSE3D(H2_label - H2_est)
        X2_mse = MSE3D(X2_label - X2_est)
        X2_ber = BER3D(X2_label, X2_est)
        if self.if_print:
            print('Iteration 1, H2 MSE', mse.mean(), 'H2 var', var_H2.mean())
            print('X2 MSE', X2_mse.mean(), 'X2 var', var_X2.mean(), 'X2 BER', X2_ber.mean())
        write_message.extend([mse.mean().item(), MSEre(mse), var_H2.mean().item(), MSEre(var_H2)])
        write_message.extend([X2_mse.mean().item(), MSEre(X2_mse), var_X2.mean().item(), MSEre(var_X2),
                              X2_ber.mean().item(), MSEre(X2_ber)])
        # loss = loss + mse / mse.detach()
        for i in range(self.unfolding_depth - 1):
            H1_est_mapped = self.real2complex(self.layers[4*i+3].forward(self.complex2real(H2_est)))
            if self.is_unfolding:
                var_H1 = None
                H1_est = (H1_est - H1_est_mapped) * self.damping_params1[i] + H1_est_mapped
            else:
                var_H1_mapped = self.var_mapping2(var_H2)
                H1_est = H1_est_mapped
                var_H1 = var_H1_mapped
                # H1_est = (H1_est + H1_est_mapped) * 0.5
                # var_H1 = var_H1 * var_H1_mapped / (var_H1 + var_H1_mapped)
            dist = MSE3D(H1_label - H1_est).view(-1, 1, 1)
            if self.if_print:
                print('Iteration ', i + 2, ', H1 MSE', dist.mean(), 'H1 var', var_H1.mean())
            write_message.extend([dist.mean().item(), MSEre(dist), var_H1.mean().item(), MSEre(var_H1)])

            H1_est, X1_est, var_X1, var_H1 = self.layers[4*i+4].forward(H1_est, X1_est, var_X1, dist,
                                                                        Y1, Np, X1_label, H1_label)
            # H1_est, X1_est, var_X1, var_H1 = self.layers[4*i+4].forward(H1_est, X1_est, var_X1, var_H1, Y1,
            #                                                             Np, X1_label, H1_label)
            H1_mse = MSE3D(H1_label - H1_est)
            X1_mse = MSE3D(X1_label - X1_est)
            X1_ber = BER3D(X1_label, X1_est)
            if self.if_print:
                print('Iteration 1, H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
                print('X1 MSE', 'X1 var', 'X1 BER')
            write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
            write_message.extend([X1_mse.mean().item(), MSEre(X1_mse), var_X1.mean().item(), MSEre(var_X1),
                                  X1_ber.mean().item(), MSEre(X1_ber)])

            # loss = loss + mse / mse.detach()
            H2_est_mapped = self.real2complex(self.layers[4*i+5].forward(self.complex2real(H1_est)))
            if self.is_unfolding:
                var_H2 = None
                H2_est = (H2_est - H2_est_mapped) * self.damping_params2[i] + H2_est_mapped
            else:
                var_H2_mapped = self.var_mapping1(var_H1)
                H2_est = H2_est_mapped
                var_H2 = var_H2_mapped
                # H2_est = (H2_est + H2_est_mapped) * 0.5
                # var_H2 = var_H2 * var_H2_mapped / (var_H2 + var_H2_mapped)
            dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
            if self.if_print:
                print('Iteration ', i + 2, ', H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
            write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])

            H2_est, X2_est, var_X2, var_H2 = self.layers[4*i+6].forward(H2_est, X2_est, var_X2, dist,
                                                                        Y2, Nd, X2_label, H2_label)
            # H2_est, X2_est, var_X2, var_H2 = self.layers[4*i+6].forward(H2_est, X2_est, var_X2, var_H2,
            #                                                             Y2, Nd, X2_label, H2_label)
            mse = MSE3D(H2_label - H2_est)
            X2_mse = MSE3D(X2_label - X2_est)
            X2_ber = BER3D(X2_label, X2_est)
            if self.if_print:
                print('Iteration ', i + 2, ', H2 MSE', mse.mean(), 'H2 var', var_H2.mean())
                print('X2 MSE', X2_mse.mean(), 'X2 var', var_X2.mean(), 'X2 BER', X2_ber.mean())
            write_message.extend([mse.mean().item(), MSEre(mse), var_H2.mean().item(), MSEre(var_H2)])
            write_message.extend([X2_mse.mean().item(), MSEre(X2_mse), var_X2.mean().item(), MSEre(var_X2),
                                  X2_ber.mean().item(), MSEre(X2_ber)])
            # loss = loss + mse / mse.detach()
        if self.writer is not None:
            self.writer.writerow(write_message)
        return loss


class VI_module(nn.Module):
    def __init__(self, basic_module: nn.Module, n_samples):
        super().__init__()
        self.basic_module = basic_module
        self.n_samples = n_samples

    def forward(self, X, X_var):
        X = X.repeat([self.n_samples, 1, 1, 1])
        X_var = X_var.repeat([self.n_samples, 1, 1, 1])
        X_samples = X + torch.randn_like(X) * X_var
        # print(X_samples.shape)
        X_outsamples = self.basic_module.forward(X_samples)
        X_outsamples = rearrange(X_outsamples, '(m u) ant car n -> m u ant car n', m=self.n_samples)
        X_pred = torch.mean(X_outsamples, dim=0, keepdim=True)
        X_var_pred = X_outsamples - X_pred
        X_var_pred = torch.sum(X_var_pred * X_var_pred, dim=0, keepdim=False) / (self.n_samples - 1)
        return X_pred.squeeze(0), X_var_pred


class Bidirection_VI(nn.Module):
    def __init__(self, cp, c, m, n, kp, k, NN_depth, unfolding_depth, alpha_init, beta_init, gamma_init,
                 eta_init, phi, demod, demod_hard, demod_H, n_layer, n_samples,
                 writer=None, if_print=False):
        super().__init__()
        self.unfolding_depth = unfolding_depth
        self.is_unfolding = False
        self.layers = nn.ModuleList([])
        self.n_samples = n_samples
        self.if_print = if_print
        self.cp = cp
        cd = c - cp
        self.cd = cd
        multiuser = (m > 1)
        for i in range(unfolding_depth-1):
            mod = get_unfolding_model(cp, cp, m, n, kp, k, alpha_init[i*2], beta_init[i*2], gamma_init[i*2],
                                      eta_init[i*2], phi, demod, demod_hard, demod_H, n_layer,
                                      writer=writer, if_print=if_print, isest1=True, init_input_var=None, multiuser=multiuser)
            self.layers.append(mod)
            mod = get_mixer_net(n, cp, n, cd, NN_depth)
            init_mapping_net(mod, is_mapping1=True)
            self.layers.append(VI_module(mod, n_samples))
            mod = get_unfolding_model(0, cd, m, n, 0, k, alpha_init[i*2+1], beta_init[i*2+1], gamma_init[i*2+1],
                                      eta_init[i*2+1], phi, demod, demod_hard, demod_H, n_layer,
                                      writer=writer, if_print=if_print, isest1=False, init_input_var=None, multiuser=multiuser)
            self.layers.append(mod)
            mod = get_mixer_net(n, cd, n, cp, NN_depth)
            init_mapping_net(mod, is_mapping1=False)
            self.layers.append(VI_module(mod, n_samples))
        mod = get_unfolding_model(cp, cp, m, n, kp, k, alpha_init[unfolding_depth*2-2],
                                  beta_init[unfolding_depth*2-2], gamma_init[unfolding_depth*2-2],
                                  eta_init[unfolding_depth*2-2], phi, demod, demod_hard, demod_H, n_layer,
                                  writer=writer, if_print=if_print, isest1=True, init_input_var=None, multiuser=multiuser)
        self.layers.append(mod)
        mod = get_mixer_net(n, cp, n, cd, NN_depth)
        init_mapping_net(mod, is_mapping1=True)
        self.layers.append(VI_module(mod, n_samples))
        mod = get_unfolding_model(0, cd, m, n, 0, k, alpha_init[unfolding_depth*2-1],
                                  beta_init[unfolding_depth*2-1], gamma_init[unfolding_depth*2-1],
                                  eta_init[unfolding_depth*2-1], phi, demod, demod_hard, demod_H, n_layer,
                                  writer=writer, if_print=if_print, isest1=False,
                                  init_input_var=None, multiuser=multiuser)
        self.layers.append(mod)

    @staticmethod
    def complex2real(H, var_H):
        H = rearrange(H, 'car ant (u n) -> u ant car n', n=1)
        var_H = rearrange(var_H, 'car ant (u n) -> u ant car n', n=1) * 0.5
        H = torch.cat((H.real, H.imag), dim=3)
        return H, var_H

    @staticmethod
    def real2complex(H, var_H):
        H_complex = H[:, :, :, 0] + 1j * H[:, :, :, 1]
        var_H = var_H[:, :, :, 0] + var_H[:, :, :, 1]
        H_complex = rearrange(H_complex, 'u ant car -> car ant u')
        var_H = rearrange(var_H, 'u ant car -> car ant u')
        return H_complex, var_H

    def forward(self, H1_est, X1_est, var_X1, var_H1, X2_est, var_X2,
                X1_label, X2_label, H1_label, H2_label, Y1, Y2, N0):
        # m: Nue, n: Nant
        # X: c x m x k
        # H_pri: c x n x m
        # Y: c x n x k
        Np = N0.repeat([1, self.cp, 1, 1]).reshape(-1, 1, 1)
        Nd = N0.repeat([1, self.cd, 1, 1]).reshape(-1, 1, 1)
        H_ener = torch.sum(H2_label.real * H2_label.real + H2_label.imag * H2_label.imag)
        loss = 0
        H1_est, X1_est, var_X1, var_H1 = self.layers[0].forward(H1_est, X1_est, var_X1, var_H1, Y1,
                                                                Np, X1_label, H1_label)
        mse = MSE(H1_label - H1_est)
        if self.if_print:
            print('Iteration 1, H1 MSE', mse, 'H1 var', var_H1.mean())
            print('X1 MSE', 'X1 var', 'X1 BER')
        print(H1_est.shape, var_H1.shape, X1_est.shape, var_X1.shape)
        loss = loss + mse / mse.detach()
        est, var = self.complex2real(H1_est, var_H1)
        est, var = self.layers[1].forward(est, var)
        H2_est, var_H2 = self.real2complex(est, var)
        print(H2_est.shape, var_H2.shape, X2_est.shape, var_X2.shape)
        # print(H2_label.shape, H2_est.shape, var_H2.shape)
        if self.if_print:
            print('Iteration 1, H2 MSE', MSE(H2_label - H2_est), 'H2 var', var_H2.mean())
        H2_est, X2_est, var_X2, var_H2 = self.layers[2].forward(H2_est, X2_est, var_X2, var_H2,
                                                                Y2, Nd, X2_label, H2_label)
        mse = MSE(H2_label - H2_est)
        if self.if_print:
            print('Iteration 1, H2 MSE', mse, 'H2 var', var_H2.mean())
            print('X2 MSE', MSE(X2_label - X2_est), 'X2 var', var_X2.mean(), 'X2 BER', BER(X2_label, X2_est))
        print(H2_est.shape, var_H2.shape, X2_est.shape, var_X2.shape)
        loss = loss + mse / mse.detach()
        for i in range(self.unfolding_depth - 1):
            est, var = self.complex2real(H2_est, var_H2)
            est, var = self.layers[4*i+3].forward(est, var)
            H1_est, var_H1 = self.real2complex(est, var)
            if self.if_print:
                print('Iteration ', i+2, ', H1 MSE', MSE(H1_label - H1_est), 'H1 var', var_H1.mean())
            # H1_est_mapped, var_H1_mapped = self.real2complex(est, var)
            # H1_est = (H1_est + H1_est_mapped) * 0.5
            # var_H1 = var_H2 * var_H1_mapped / (var_H1 + var_H1_mapped)
            H1_est, X1_est, var_X1, var_H1 = self.layers[4*i+4].forward(H1_est, X1_est, var_X1, var_H1,
                                                                        Y1, Np, X1_label, H1_label)
            mse = MSE(H1_label - H1_est)
            if self.if_print:
                print('Iteration ', i+2, ', H1 MSE', mse, 'H1 var', var_H1.mean())
                print('X1 MSE', 'X1 var', 'X1 BER')
            loss = loss + mse / mse.detach()
            est, var = self.complex2real(H1_est, var_H1)
            est, var = self.layers[4*i+5].forward(est, var)
            H2_est, var_H2 = self.real2complex(est, var)
            # H2_est_mapped, var_H2_mapped = self.real2complex(est, var)
            # H2_est = (H2_est + H2_est_mapped) * 0.5
            # var_H2 = var_H2 * var_H2_mapped / (var_H2 + var_H2_mapped)
            if self.if_print:
                print('Iteration ', i+2, ', H2 MSE', MSE(H2_label - H2_est), 'H2 var', var_H2.mean())
            H2_est, X2_est, var_X2, var_H2 = self.layers[4*i+6].forward(H2_est, X2_est, var_X2, var_H2,
                                                                        Y2, Nd, X2_label, H2_label)
            mse = MSE(H2_label - H2_est)
            if self.if_print:
                print('Iteration ', i+2, ', H2 MSE', mse, 'H2 var', var_H2.mean())
                print('X2 MSE', MSE(X2_label - X2_est), 'X2 var', var_X2.mean(), 'X2 BER', BER(X2_label, X2_est))
            loss = loss + mse / mse.detach()
        return loss


def get_unfolding_model(cp, c, m, n, kp, k, alpha_init, beta_init, gamma_init,
                        eta_init, phi, demod, demod_hard, demod_H, n_layer, bs=1,
                        writer=None, if_print=False, isest1=True, init_input_var=None,
                        multiuser=False, first_data_layer=False):
    if multiuser:
        return BiGaBP_unfolding(cp, c, m, n, kp, k, alpha_init, beta_init, gamma_init,
                                eta_init, phi, demod, demod_hard, demod_H, n_layer, bs,
                                writer, if_print, isest1, init_input_var, first_data_layer=first_data_layer)
    elif isest1:
        return Est_pilot()
    else:
        return Est_data()

