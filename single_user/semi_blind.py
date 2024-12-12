import torch.nn as nn
import torch
from einops import rearrange

# from basic_modules import get_unfolding_model, MSE
from model.model import get_mixer_net

REAL_VAR = False
BEST_DAMPING = False
DAMPING = 0.
UNFOLDING = True
GAMMA = 0.5

MSE = lambda H: torch.mean(H.real * H.real + H.imag * H.imag)
MSEre = lambda H: torch.mean(H * H).item()
MSE3D = lambda H: torch.mean(H.real * H.real + H.imag * H.imag, dim=[1, 2])
MSE4D = lambda H: torch.mean(H.real * H.real + H.imag * H.imag, dim=[1, 2, 3])


# Only for QPSK modulation
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


def init_mapping_net(model, is_mapping1=True):
    if is_mapping1:
        model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_8ant1.pth'))
        # model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_4ant_4car1.pth'))
    else:
        model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_8ant2.pth'))
        # model.load_state_dict(torch.load('models/mixer_standard_32ant_64car_300k_white_7dB_4ant_4car2.pth'))


def demod_QPSK(u, v):
    def demod(u, v, p1):
        # v = v.clamp(min=1e-5, max=None)
        p_1 = p1 / (p1 + (1-p1) * torch.exp(-2 * u.div(v)))
        u_post = 2 * p_1 - 1
        v_post = torch.mean(1 - u_post.pow(2), dim=1, keepdim=True)
        # p_1 = p1 / (p1 + (1-p1) * torch.exp(-2 * u.div(v.unsqueeze(2).unsqueeze(3))))
        # u_post = 2 * p_1 - 1
        # v_post = torch.mean(1 - u_post.pow(2), dim=2).squeeze(2)
        # u_post = u_tmp.mul(v)
        #     u_tmp.mul(v.gt(1e-5).unsqueeze(2).unsqueeze(3)) +\
        #          u.mul(v.le(1e-5).unsqueeze(2).unsqueeze(3))
        # v_post = v_tmp.mul(v.gt(1e-5))
        return u_post, v_post

    u_post_r, v_post_r = demod(1.41421356 * u.real, v, 0.5)
    u_post_i, v_post_i = demod(1.41421356 * u.imag, v, 0.5)
    return (u_post_r + 1j * u_post_i)/1.41421356, (v_post_r + v_post_i)/2


def merge_est(est1, var1, est2, var2):
    var = var1 * var2 / (var1 + var2)
    est = est1 / var1 + est2 / var2
    return est * var, var


class Est_data(nn.Module):
    def __init__(self, is_unfolding=False):
        # Xp: c x 1 x k
        # H: c x ant x 1
        # Y: c x ant x k
        super().__init__()
        self.is_unfolding = is_unfolding

    def forward(self, H, X_est, var_X, H_var, Y, N0):  # , H_label, X_label):
        # H: c x ant x 1
        # H_var: c x 1 x 1
        # if self.init_var is not None:
        #     H_var = self.init_var
        H_var = torch.clamp(H_var, min=0.001, max=None)
        H_ener = torch.sum(H.real * H.real + H.imag * H.imag, dim=[1, 2], keepdim=True).detach()
        X_est = (H.mH @ Y) / H_ener
        X_var = (H_var + N0) / H_ener
        X_est, X_var = demod_QPSK(X_est, X_var.detach())
        H_newpred = Y * X_est.conj()
        H_newvar = H_ener * X_var + N0
        if BEST_DAMPING:
            H_newpred, H_newvar = merge_est(H, H_var, H_newpred, H_newvar)
        elif DAMPING > 0 and not self.is_unfolding:
            H_newpred = H_newpred * (1 - DAMPING) + H * DAMPING
            H_newvar = H_newvar * (1 - DAMPING) * (1 - DAMPING) + H_var * DAMPING * DAMPING
        return H_newpred, X_est, X_var, H_newvar


class Est_pilot(nn.Module):
    def __init__(self, is_unfolding=False):
        # Xp: c x 1 x k
        # H: c x ant x 1
        # Y: c x ant x k
        super().__init__()

    def forward(self, H, X_est, var_X, H_var, Y, Xp, N0):
        # H: c x ant x 1
        # H_var: c x 1 x 1
        H_newpred = Y * Xp.conj()
        H_newvar = N0
        # if self.init_var is not None:
        #     H, H_var = merge_est(H, self.init_var, H_newpred, H_newvar)
        #     return H, None, None, torch.zeros(1)
        # el
        if H_var is not None:
            H, H_var = merge_est(H, H_var, H_newpred, H_newvar)
            return H, None, None, H_var
        else:
            return H_newpred, None, None, H_newvar


class Unfolding_NN(nn.Module):
    def __init__(self, batch_size, cp, c, m, n, kp, k, NN_depth, unfolding_depth, N0, alpha_init, beta_init, gamma_init,
                 eta_init, phi, demod, demod_hard, demod_H, n_layer,
                 writer=None, if_print=False, init_input_var=None, var_mapping1=lambda x: x, var_mapping2=lambda x: x):
        super().__init__()
        self.batch_size = batch_size
        self.unfolding_depth = unfolding_depth
        self.var_mapping1 = var_mapping1
        self.var_mapping2 = var_mapping2
        self.if_print = if_print
        if init_input_var is None:
            self.is_unfolding = False
            init_input_var = [None] * (2 * unfolding_depth)
        else:
            self.is_unfolding = True
            self.input_var = nn.Parameter(torch.tensor(init_input_var))
            self.damping_params1 = nn.Parameter(torch.ones(unfolding_depth-1) * 0.5)
            self.damping_params2 = nn.Parameter(torch.ones(unfolding_depth-1) * 0.5)
            self.unfolding_params = [self.input_var, self.damping_params1, self.damping_params2]
        self.layers = nn.ModuleList([])
        cd = c - cp
        self.cp = cp
        self.cd = cd
        self.writer = writer
        for i in range(unfolding_depth-1):
            mod = get_unfolding_model(cp, cp, m, n, kp, k, N0, alpha_init, beta_init, gamma_init,
                                      eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                      writer, isest1=True, is_unfolding=self.is_unfolding)
            self.layers.append(mod)
            mod = get_mixer_net(n, cp, n, cd, NN_depth)
            init_mapping_net(mod, is_mapping1=True)
            self.layers.append(mod)
            mod = get_unfolding_model(0, cd, m, n, 0, k, N0, alpha_init, beta_init, gamma_init,
                                      eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                      writer, isest1=False, is_unfolding=self.is_unfolding)
            self.layers.append(mod)
            mod = get_mixer_net(n, cd, n, cp, NN_depth)
            init_mapping_net(mod, is_mapping1=False)
            self.layers.append(mod)
        mod = get_unfolding_model(cp, cp, m, n, kp, k, N0, alpha_init, beta_init, gamma_init,
                                  eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                  writer, isest1=True, is_unfolding=self.is_unfolding)
        self.layers.append(mod)
        mod = get_mixer_net(n, cp, n, cd, NN_depth)
        init_mapping_net(mod, is_mapping1=True)
        self.layers.append(mod)
        mod = get_unfolding_model(0, cd, m, n, 0, k, N0, alpha_init, beta_init, gamma_init,
                                  eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                  writer, isest1=False, is_unfolding=self.is_unfolding)
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

    def complex2real(self, H):
        H = rearrange(H, '(bs car) ant (u n) -> (bs u) ant car n', bs=self.batch_size, n=1)
        # print(H.shape)
        H = torch.cat((H.real, H.imag), dim=3)
        # print(H.shape)
        return H

    def real2complex(self, H):
        # print(H.shape)
        H_complex = H[:, :, :, 0] + 1j * H[:, :, :, 1]
        H_complex = rearrange(H_complex, '(bs u) ant car -> (bs car) ant u', bs=self.batch_size)
        # print(H_complex.shape)
        return H_complex

    def forward(self, H1_est, X1_est, var_X1, var_H1, X2_est, var_X2,
                X1_label, X2_label, H1_label, H2_label, Y1, Y2, N0):
        # m: Nue, n: Nant
        # X: c x m x k
        # H_pri: c x n x m
        # Y: c x n x k
        write_message = []
        Np = N0.repeat([1, self.cp, 1, 1]).reshape(-1, 1, 1)
        Nd = N0.repeat([1, self.cd, 1, 1]).reshape(-1, 1, 1)
        loss = 0
        H1_est, X1_est, var_X1, var_H1 = self.layers[0].forward(H1_est, X1_est, var_X1, var_H1, Y1, X1_label, Np)
        H1_mse = MSE3D(H1_label - H1_est)
        if self.if_print:
            print('Iteration 1, H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
            print('X1 MSE', 'X1 var', 'X1 BER')
        write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
        # loss = loss + torch.sum(H1_mse / (H1_mse.detach() + 0.1))

        H1 = self.complex2real(H1_est)
        H2 = self.layers[1].forward(H1.detach())
        H2_est = self.real2complex(H2)
        if self.is_unfolding:
            var_H2 = self.var_mapping1(var_H1, self.input_var[0])
            # print(var_H1.mean(), self.input_var[0])
        else:
            var_H2 = self.var_mapping1(var_H1)
        dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
        if self.if_print:
            print('Iteration 1, H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
        write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])
        loss2 = torch.mean((var_H2 - dist.detach()).pow(2))
        loss = loss + GAMMA * loss2

        H2_est, X2_est, var_X2, var_H2 = self.layers[2].forward(H2_est.detach(), X2_est, var_X2.detach(),
                                                                dist if REAL_VAR else var_H2, Y2, Nd)
        mse = MSE3D(H2_label - H2_est)
        X2_mse = MSE3D(X2_label - X2_est)
        X2_ber = BER3D(X2_label, X2_est)
        if self.if_print:
            print('Iteration 1, H2 MSE', mse.mean(), 'H2 var', var_H2.mean())
            print('X2 MSE', X2_mse.mean(), 'X2 var', var_X2.mean(), 'X2 BER', X2_ber.mean())
        write_message.extend([mse.mean().item(), MSEre(mse), var_H2.mean().item(), MSEre(var_H2)])
        write_message.extend([X2_mse.mean().item(), MSEre(X2_mse), var_X2.mean().item(), MSEre(var_X2),
                              X2_ber.mean().item(), MSEre(X2_ber)])
        for i in range(self.unfolding_depth - 1):
            with torch.no_grad():
                H1_est_mapped = self.real2complex(self.layers[4*i+3].forward(self.complex2real(H2_est)))
            if self.is_unfolding:
                # var_H1 = torch.zeros(1)
                H1_est = (H1_est - H1_est_mapped) * self.damping_params1[i] + H1_est_mapped
                var_H1 = self.var_mapping2(var_H2, self.input_var[i*2+1]) * (1 - self.damping_params1[i].detach()) \
                         + var_H1 * self.damping_params1[i].detach()
            else:
                var_H1_mapped = self.var_mapping2(var_H2)
                H1_est = H1_est_mapped
                var_H1 = var_H1_mapped
                # H1_est = (H1_est + H1_est_mapped) * 0.5
                # var_H1 = var_H1 * var_H1_mapped / (var_H1 + var_H1_mapped)
            dist = MSE3D(H1_label - H1_est).view(-1, 1, 1)
            if self.if_print:
                print('Iteration ', i+2, ', H1 MSE', dist.mean(), 'H1 var', var_H1.mean())
            loss2 = (var_H1 - dist.detach()).pow(2)
            loss2 = torch.mean(loss2)
            loss1 = torch.mean(dist)
            #loss2 = torch.mean(loss2 / (loss2.detach() + 0.1))
            #loss1 = torch.mean(dist / (dist.detach() + 0.1))
            loss = loss + loss1.pow(2) + GAMMA * loss2
            write_message.extend([dist.mean().item(), MSEre(dist), var_H1.mean().item(), MSEre(var_H1)])
            #print(var_H1.mean().item(), dist.mean().item())
            '''
            H1_est, X1_est, var_X1, var_H1 = self.layers[4*i+4].forward(H1_est, X1_est, var_X1,
                                                                        dist if REAL_VAR else var_H1,
                                                                        Y1, X1_label, Np)
            '''
            H1_mse = MSE3D(H1_label - H1_est)
            if self.if_print:
                print('Iteration', i+2, ', H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
                print('X1 MSE', 'X1 var', 'X1 BER')
            write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
            with torch.no_grad():
                H2_est_mapped = self.real2complex(self.layers[4*i+5].forward(self.complex2real(H1_est)))
            if self.is_unfolding:
                # var_H2 = torch.zeros(1)
                H2_est = (H2_est - H2_est_mapped) * self.damping_params2[i] + H2_est_mapped
                with torch.no_grad():
                    dd = self.damping_params2[i]
                var_H2 = self.var_mapping1(var_H1, self.input_var[i*2+2]) * (1 - dd) \
                         + var_H2 * dd
                #print(var_H1.mean(), var_H2.mean())
                #print(H1_mse.mean())
                #print(X2_ber.mean())
            else:
                var_H2_mapped = self.var_mapping1(var_H1)
                H2_est = H2_est_mapped
                var_H2 = var_H2_mapped
                # H2_est = (H2_est + H2_est_mapped) * 0.5
                # var_H2 = var_H2 * var_H2_mapped / (var_H2 + var_H2_mapped)
            dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
            #print(dist.mean())
            if self.if_print:
                print('Iteration ', i+2, ', H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
            write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])
            loss2 = (var_H2 - dist.detach()).pow(2)
            #print(loss2.mean())
            #loss2 = torch.mean(loss2 / (loss2.detach() + 0.1))
            #loss1 = torch.mean(dist / (dist.detach() + 0.1))
            #loss = loss + loss1 + GAMMA * loss2
            loss = loss + torch.mean(dist).pow(2) + GAMMA * torch.mean(loss2)
            #print(var_H2.mean().item(), dist.mean().item())
            '''
            H2_est, X2_est, var_X2, var_H2 = self.layers[4*i+6].forward(H2_est.detach(), X2_est, var_X2,
                                                                        dist if REAL_VAR else var_H2, Y2, Nd)
            '''
            mse = MSE3D(H2_label - H2_est)
            X2_mse = MSE3D(X2_label - X2_est)
            X2_ber = BER3D(X2_label, X2_est)
            if self.if_print:
                print('Iteration ', i+2, ', H2 MSE', mse.mean(), 'H2 var', var_H2.mean())
                print('X2 MSE', X2_mse.mean(), 'X2 var', var_X2.mean(), 'X2 BER', X2_ber.mean())
            write_message.extend([mse.mean().item(), MSEre(mse), var_H2.mean().item(), MSEre(var_H2)])
            write_message.extend([X2_mse.mean().item(), MSEre(X2_mse), var_X2.mean().item(), MSEre(var_X2),
                                  X2_ber.mean().item(), MSEre(X2_ber)])
        if self.writer is not None:
            self.writer.writerow(write_message)
        return loss, H1_mse.mean(), mse.mean(), MSE(X2_label - X2_est), BER(X2_label, X2_est), write_message


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
    def __init__(self, cp, c, m, n, kp, k, NN_depth, unfolding_depth, N0, alpha_init, beta_init, gamma_init,
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
        for i in range(unfolding_depth-1):
            mod = get_unfolding_model(cp, cp, m, n, kp, k, N0, alpha_init, beta_init, gamma_init,
                                      eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                      writer, isest1=True, init_input_var=None)
            self.layers.append(mod)
            mod = get_mixer_net(n, cp, n, cd, NN_depth)
            init_mapping_net(mod, is_mapping1=True)
            self.layers.append(VI_module(mod, n_samples))
            mod = get_unfolding_model(0, cd, m, n, 0, k, N0, alpha_init, beta_init, gamma_init,
                                      eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                      writer, isest1=False, init_input_var=None)
            self.layers.append(mod)
            mod = get_mixer_net(n, cd, n, cp, NN_depth)
            init_mapping_net(mod, is_mapping1=False)
            self.layers.append(VI_module(mod, n_samples))
        mod = get_unfolding_model(cp, cp, m, n, kp, k, N0, alpha_init, beta_init, gamma_init,
                                  eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                  writer, isest1=True, init_input_var=None)
        self.layers.append(mod)
        mod = get_mixer_net(n, cp, n, cd, NN_depth)
        init_mapping_net(mod, is_mapping1=True)
        self.layers.append(VI_module(mod, n_samples))
        mod = get_unfolding_model(0, cd, m, n, 0, k, N0, alpha_init, beta_init, gamma_init,
                                  eta_init, phi, demod, demod_hard, demod_H, n_layer,
                                  writer, isest1=False, init_input_var=None)
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
        # H_ener = torch.sum(H2_label.real * H2_label.real + H2_label.imag * H2_label.imag)
        # print(H_ener)
        loss = 0
        write_message = []
        H1_est, X1_est, var_X1, var_H1 = self.layers[0].forward(H1_est, X1_est, var_X1, var_H1, Y1, X1_label, Np)
        H1_mse = MSE3D(H1_label - H1_est)
        if self.if_print:
            print('Iteration 1, H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
            print('X1 MSE', 'X1 var', 'X1 BER')
        write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
        # loss = loss + H1_mse / H1_mse.detach()

        est, var = self.complex2real(H1_est, var_H1)
        est, var = self.layers[1].forward(est, var)
        H2_est, var_H2 = self.real2complex(est, var)
        # print(H2_label.shape, H2_est.shape, var_H2.shape)

        dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
        if self.if_print:
            print('Iteration 1, H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
        write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])

        H2_est, X2_est, var_X2, var_H2 = self.layers[2].forward(H2_est, X2_est, var_X2,
                                                                dist if REAL_VAR else var_H2, Y2, Nd)
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
            est, var = self.complex2real(H2_est, var_H2)
            est, var = self.layers[4*i+3].forward(est, var)
            H1_est, var_H1 = self.real2complex(est, var)

            dist = MSE3D(H1_label - H1_est).view(-1, 1, 1)
            if self.if_print:
                print('Iteration ', i + 2, ', H1 MSE', dist.mean(), 'H1 var', var_H1.mean())
            write_message.extend([dist.mean().item(), MSEre(dist), var_H1.mean().item(), MSEre(var_H1)])
            # H1_est_mapped, var_H1_mapped = self.real2complex(est, var)
            # H1_est = (H1_est + H1_est_mapped) * 0.5
            # var_H1 = var_H2 * var_H1_mapped / (var_H1 + var_H1_mapped)
            H1_est, X1_est, var_X1, var_H1 = self.layers[4*i+4].forward(H1_est, X1_est, var_X1, var_H1,
                                                                        Y1, X1_label, Np)
            H1_mse = MSE3D(H1_label - H1_est)
            if self.if_print:
                print('Iteration 1, H1 MSE', H1_mse.mean(), 'H1 var', var_H1.mean())
                print('X1 MSE', 'X1 var', 'X1 BER')
            write_message.extend([H1_mse.mean().item(), MSEre(H1_mse), var_H1.mean().item(), MSEre(var_H1)])
            # loss = loss + H1_mse / H1_mse.detach()

            est, var = self.complex2real(H1_est, var_H1)
            est, var = self.layers[4*i+5].forward(est, var)
            H2_est, var_H2 = self.real2complex(est, var)
            # H2_est_mapped, var_H2_mapped = self.real2complex(est, var)
            # H2_est = (H2_est + H2_est_mapped) * 0.5
            # var_H2 = var_H2 * var_H2_mapped / (var_H2 + var_H2_mapped)
            dist = MSE3D(H2_label - H2_est).view(-1, 1, 1)
            if self.if_print:
                print('Iteration ', i + 2, ', H2 MSE', dist.mean(), 'H2 var', var_H2.mean())
            write_message.extend([dist.mean().item(), MSEre(dist), var_H2.mean().item(), MSEre(var_H2)])
            H2_est, X2_est, var_X2, var_H2 = self.layers[4*i+6].forward(H2_est, X2_est, var_X2, var_H2, Y2, Nd)
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
        return loss, H1_mse, mse, MSE(X2_label - X2_est), BER(X2_label, X2_est)


def get_unfolding_model(cp, c, m, n, kp, k, N0, alpha_init, beta_init, gamma_init,
                        eta_init, phi, demod, demod_hard, demod_H, n_layer, writer,
                        isest1=False, is_unfolding=False):
    # X_p = X_label[:, :cp, :, :]
    # X_d = X_label[:, cp:, :, :]
    # N0 = N0.repeat([1, c, 1, 1]).reshape(-1, 1, 1)
    if isest1:
        return Est_pilot(is_unfolding)
    else:
        return Est_data(is_unfolding)
