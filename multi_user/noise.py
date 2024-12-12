import torch
import math
from einops import rearrange


# ener: E(n^Hn)/(2N)
class Gen_noise:
    def __init__(self, ener, device, is_mse):
        self.is_mse = is_mse
        self.ener = ener
        self.device = device

    @staticmethod
    def calc_ener(H):
        return torch.sqrt(torch.mean(H * H, dim=[1, 2, 3], keepdim=True))

    def generate_noise(self, H):
        raise NotImplementedError

    def add_noise(self, H):
        return H + self.generate_noise(H)


class White(Gen_noise):
    def __init__(self, ener, device, is_mse, power_distribution_name):
        super().__init__(ener, device, is_mse)
        if power_distribution_name == 'invariant':
            self.power_generator = lambda x, b: x
        elif power_distribution_name == 'uniform':
            self.power_generator = lambda x, b: torch.rand(b, 1, 1, 1, device=device) * x * 2
        else:
            raise NotImplementedError

    def generate_noise(self, H):
        noise = torch.randn_like(H)
        b = H.shape[0]
        if self.is_mse:
            return noise * math.sqrt(self.power_generator(self.ener, b))
        else:
            return noise * torch.sqrt(self.power_generator(self.ener, b)) * self.calc_ener(H)


class Path(Gen_noise):
    def __init__(self, ener, device, is_mse, antenna_shape, num_path, variant_path, variant_power):
        super().__init__(ener, device, is_mse)
        self.antenna_shape = antenna_shape
        self.num_path = num_path
        self.variant_path = variant_path
        self.variant_power = variant_power

    def get_basic_noise(self, total_path, num_car):
        if len(self.antenna_shape) == 1:
            basic_vector_ant = torch.arange(self.antenna_shape[0], device=self.device).view(1, self.antenna_shape[0], 1) * 6.28
            rand_theta_ant = torch.rand(total_path, 1, 1, device=self.device)
            theta = rand_theta_ant * basic_vector_ant
        else:
            basic_vector_ant1 = torch.arange(self.antenna_shape[0], device=self.device).view(1, self.antenna_shape[0], 1, 1) * 6.28
            basic_vector_ant2 = torch.arange(self.antenna_shape[1], device=self.device).view(1, 1, self.antenna_shape[1], 1) * 6.28
            rand_theta_ant1 = torch.rand(total_path, 1, 1, 1, device=self.device)
            rand_theta_ant2 = torch.rand(total_path, 1, 1, 1, device=self.device)
            theta = rand_theta_ant1 * basic_vector_ant1 + rand_theta_ant2 * basic_vector_ant2
            theta = theta.view(1, -1, 1)
        basic_vector_car = torch.arange(num_car, device=self.device).view(1, 1, num_car) * 6.28
        rand_theta_car = torch.rand(total_path, 1, 1, device=self.device)
        theta = theta + rand_theta_car * basic_vector_car
        path_gain = torch.randn(total_path, 1, 1, dtype=torch.complex64, device=self.device)
        basic_noise = torch.exp(1j * theta) * path_gain
        return torch.stack((basic_noise.real, basic_noise.imag), dim=3), path_gain

    def generate_noise(self, H):
        s = list(H.shape)
        b = s[0]
        c = s[2]
        if not self.variant_path:
            basic_noise, path_gain = self.get_basic_noise(self.num_path * b, c)
            noise = rearrange(basic_noise, '(n b) ant car c -> n b ant car c', n=self.num_path)
            noise = torch.sum(noise, dim=0)
            if self.variant_power:
                noise = noise * (self.ener * 2 / self.num_path)
            else:
                path_power = path_gain.real * path_gain.real + path_gain.imag * path_gain.imag
                path_power = path_power.view(self.num_path, b, 1, 1, 1)
                noise = noise * (self.ener * 2) / torch.sum(path_power, dim=0)
        else:
            paths = torch.poisson(torch.rand(b) * self.num_path)
            basic_noise, path_gain = self.get_basic_noise(int(sum(paths).item()), c)
            noise = self.sum_group(basic_noise, paths)
            if self.variant_power:
                noise = noise * (self.ener * 2 / self.num_path)
            else:
                path_power = path_gain.real * path_gain.real + path_gain.imag * path_gain.imag
                power = self.sum_group(path_power, paths)
                noise = noise * (self.ener * 2) / power
        if self.is_mse:
            return noise
        else:
            return noise * self.calc_ener(H)

    @staticmethod
    def sum_group(A, groups):
        AA = torch.split(A, groups.int().tolist(), dim=0)
        result = []
        for aa in AA:
            result.append(torch.sum(aa, dim=0))
        return torch.stack(result, dim=0)


class Doppler(Gen_noise):
    def __init__(self, device, Doppler_window_length, alpha, alpha_fixed, num_car):
        assert Doppler_window_length % 2 == 1
        self.Doppler_window_length = Doppler_window_length
        self.main_lobe_pos = Doppler_window_length // 2
        self.alpha = alpha
        self.kernel_power_mask = torch.cat((self.main_lobe_pos - torch.arange(self.main_lobe_pos + 1),
                                            torch.arange(self.main_lobe_pos)), dim=0).to(device)
        self.kernel_power_mask = self.kernel_power_mask.view(1, -1) * 0.5
        self.alpha_fixed = alpha_fixed
        if alpha_fixed:
            self.kernel_power_mask = torch.pow(alpha, self.kernel_power_mask) * 0.5
            self.kernel_power_mask[0, self.main_lobe_pos] = 0
            ener = 0
            for i in range(self.main_lobe_pos):
                ener += pow(alpha, i+1) * 2
        else:
            ener = 0
            for i in range(self.main_lobe_pos):
                ener += pow(alpha, i+1) * 2 / (i+2)
        # print('noise ener:', ener)
        super().__init__(ener, device, False)

        # Generate kernel
        mat = torch.zeros(1, num_car, num_car, Doppler_window_length)
        for i in range(num_car):
            for j in range(Doppler_window_length):
                pos = i + j - self.main_lobe_pos
                if 0 <= pos < num_car:
                    mat[0, i, pos, j] = 1
        self.mat = mat.to(self.device)

    def get_power_mask(self, b):
        rand_alpha = torch.rand(b, 1, device=self.device) * self.alpha
        mask = torch.pow(rand_alpha, self.kernel_power_mask) * 0.5
        mask[:, self.main_lobe_pos] = 0
        return mask

    def calc_conv(self, ker, H):
        # ker: b * len
        # H: b * ant * car, real
        # ker_update: b * car * car
        # output: b * ant * car * 1
        ker = ker.view(-1, 1, 1, self.Doppler_window_length)
        ker_update = torch.mul(self.mat, ker).sum(3)
        return torch.matmul(ker_update.unsqueeze(1), H.unsqueeze(3))

    def generate_conv_kernels(self, b):
        if self.alpha_fixed:
            kernel = self.kernel_power_mask
        else:
            kernel = self.get_power_mask(b)
        kernels_real = torch.randn(b, self.Doppler_window_length, device=self.device) * kernel
        kernels_imag = torch.randn(b, self.Doppler_window_length, device=self.device) * kernel
        return kernels_real, kernels_imag

    def generate_noise(self, H):
        b = H.shape[0]
        ker_real, ker_imag = self.generate_conv_kernels(b)
        noise_real = self.calc_conv(ker_real, H[:, :, :, 0]) - self.calc_conv(ker_imag, H[:, :, :, 1])
        noise_imag = self.calc_conv(ker_imag, H[:, :, :, 0]) + self.calc_conv(ker_real, H[:, :, :, 1])
        return torch.cat((noise_real, noise_imag), dim=3)


class Noise_Adder:
    def __init__(self, additive=True):
        self.additive = additive
        if not additive:
            self.counter = 0
        self.gens = []
        self.num_gens = 0

    def add_gen(self, gen):
        self.gens.append(gen)
        self.num_gens += 1

    def add_noise(self, H):
        if self.additive:
            H_out = H
            for gen in self.gens:
                H_out = H_out + gen.generate_noise(H)
            return H_out
        else:
            H_out = H + self.gens[self.counter].generate_noise(H)
            self.counter += 1
            if self.counter == self.num_gens:
                self.counter = 0
            return H_out


def get_noise_adder(noise):
    name = noise['name']
    additive = not ('rand' in name)
    noise_adder = Noise_Adder(additive)
    if 'white' in name:
        param = noise['white']
        noise_adder.add_gen(White(param['ener'], torch.device(noise['device']),
                                  param['is_mse'], param['power_distribution_name']))
    if 'path' in name:
        param = noise['path']
        noise_adder.add_gen(Path(param['ener'], torch.device(noise['device']), param['is_mse'],
                                 param['antenna_shape'], param['num_path'],
                                 param['variant_path'], param['variant_power']))
    if 'Doppler' in name:
        param = noise['Doppler']
        noise_adder.add_gen(Doppler(torch.device(noise['device']), param['Doppler_window_length'],
                                    param['alpha'], param['alpha_fixed'], param['num_car']))
    return noise_adder
