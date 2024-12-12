import torch
import torch.nn as nn
import numpy as np
import os
import json
import csv
from model.model import get_mixer_net
from noise import get_noise_adder
from utils import get_dataset, NMSE


MSE = lambda x: torch.mean(x * x)


gpu_list = '7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
experiment_suffix = '_7dB'
np.random.seed(1)

config = {
    'input_car_size': 4,
    'ant_size': 32,
    'input_ant_size': 4,
    'car_size': 64,

    'dataset_path': '/mnt/HD2/yyz/MIMOnoisedata/',
    'dataset_name': '32ant_64car_300k',
    'ratio': [0.8, 0.2],
    'model_name': 'mixer_standard'
}
num_car = config['car_size']
num_ant = config['input_ant_size']
noise_config = {
    'name': 'white',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'white': {
        'ener': 1.0,
        'is_mse': True,
        'power_distribution_name': 'invariant'
    },
    'path': {
        'ener': 1.0,
        'is_mse': True,
        'antenna_shape': [num_ant],
        'num_path': 3,
        'variant_path': False,
        'variant_power': False
    },
    'Doppler': {
        'Doppler_window_length': 5,
        'alpha': 0.1,
        'alpha_fixed': True,
        'num_car': num_car
    }
}
config['noise'] = noise_config
experiment_name = config['model_name'] + '_' + config['dataset_name'] \
                  + '_' + config['noise']['name'] + experiment_suffix
print(config)
with open('configs/' + experiment_name + '.json', 'w') as f:
    json.dump(config, f)

# with open(name, 'r') as f:
#     config = json.load(f)

if 'mixer' in config['model_name']:
    batch_size = 256
    epochs = 200
    learning_rate = 1e-3
    if 'standard' in config['model_name']:
        depth = 5
    elif 'light' in config['model_name']:
        depth = 3
    elif 'deep' in config['model_name']:
        depth = 8
    else:
        raise NotImplementedError
    model1 = get_mixer_net(config['input_ant_size'], config['input_car_size'],
                           config['input_ant_size'], config['car_size']-config['input_car_size'], depth)

    model2 = get_mixer_net(config['input_ant_size'], config['car_size'] - config['input_car_size'],
                           config['input_ant_size'], config['input_car_size'], depth)
else:
    raise NotImplementedError
print(model1, model2)

# Dataset
datasets = get_dataset(ratio=config['ratio'], seed=1234,
                       dataset_path=config['dataset_path'], name=config['dataset_name'])

test_loader = torch.utils.data.DataLoader(
    datasets[1], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


in_ant = []
out_ant = []
in_car = []
out_car = []
for i in range(0, config['ant_size'], (config['ant_size']-1)//(config['input_ant_size']-1)):
    in_ant.append(i)
for i in range(0, config['car_size'], (config['car_size']-1)//(config['input_car_size']-1)):
    in_car.append(i)
for i in range(config['car_size']):
    if i not in in_car:
        out_car.append(i)
print(out_car)


def get_test_result(model, test_loader, noise_adder, in_ant, in_car, out_car):
    model.eval()
    sum_nmse = 0
    sum_loss = 0
    for i, input in enumerate(test_loader):
        h_full = input[0].cuda().float()
        print(in_car, in_ant)
        h_input = h_full[:, :in_ant, in_car, :]
        print(h_full.shape)
        h_input = noise_adder.add_noise(h_input)
        h_output = h_full[:, :in_ant, out_car, :]
        print(MSE(h_full[:, :in_ant, in_car, :] - h_input))
        # h_output = h_full[:, out_ant, out_car, :]
        # h_output = h_full
        with torch.no_grad():
            h_now_output = model(h_input)
            nmse = NMSE(h_output, h_now_output)
            sum_nmse += nmse
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss.item()
            sum_loss += loss
        print(MSE(h_now_output - h_output), 'output')
        if i == 10:
            exit(1)
    avg_nmse = (sum_nmse / (i + 1))
    test_loss = (sum_loss / (i + 1))
    return avg_nmse, test_loss


def test_models(model1, model2, model_name_list, noise_config_list, test_loader, writer1, writer2, in_ant, in_car, out_car):
    for model_name in model_name_list:
        all_nmse = [model_name+'NMSE']
        all_mse = [model_name+'MSE']
        model1 = model1.cuda()
        model1.load_state_dict(torch.load(model_name+'1.pth'))
        model2 = model2.cuda()
        model2.load_state_dict(torch.load(model_name+'2.pth'))
        for noise_config in noise_config_list:
            noise_adder = get_noise_adder(noise_config)
            avg_nmse, test_loss = get_test_result(model1, test_loader, noise_adder, in_ant, in_car, out_car)
            all_nmse.append(avg_nmse)
            all_mse.append(test_loss)
        writer1.writerow(all_nmse)
        writer1.writerow(all_mse)
        for noise_config in noise_config_list:
            noise_adder = get_noise_adder(noise_config)
            avg_nmse, test_loss = get_test_result(model2, test_loader, noise_adder, in_ant, out_car, incar)
            all_nmse.append(avg_nmse)
            all_mse.append(test_loss)
        writer2.writerow(all_nmse)
        writer2.writerow(all_mse)


model_name1 = 'models/mixer_standard_32ant_64car_300k_'
model_name2 = ['_7dB_4ant_4car']
model_names = ['white']
model_name_list = []
for n2 in model_name2:
    for n in model_names:
        model_name_list.append(model_name1+n+n2)
noise_config_list = []
eners = [1, 1.2, 1.4]
#eners = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 10, 15, 20]
alphas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2]
f1 = open('mapping_result_4ant_4car1.csv', 'w')
f2 = open('mapping_result_4ant_4car2.csv', 'w')
import copy
for e in eners:
    noise_config['white']['ener'] = e
    noise_config_list.append(copy.deepcopy(noise_config))
# noise_config['name'] = 'path'
# for npath in [1, 2, 3, 4, 5]:
#     noise_config['path']['num_path'] = npath
#     for e in eners:
#         noise_config['path']['ener'] = e
#         noise_config_list.append(copy.deepcopy(noise_config))
# noise_config['name'] = 'Doppler'
# for a in alphas:
#     noise_config['Doppler']['alpha'] = a
#     noise_config_list.append(copy.deepcopy(noise_config))

writer1 = csv.writer(f1)
writer1.writerow(eners)
writer1.writerow(alphas)
writer2 = csv.writer(f2)
writer2.writerow(eners)
writer2.writerow(alphas)
test_models(model1, model2, model_name_list, noise_config_list, test_loader, writer1, writer2, config['input_ant_size'], in_car, out_car)
