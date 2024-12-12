import torch
import torch.nn as nn
import numpy as np
import os
import json
import csv
from model.model import get_mixer_net
from noise import get_noise_adder
from utils import get_dataset, NMSE, cosine_similarity


MSE = lambda x: torch.mean(x * x)


gpu_list = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
experiment_suffix = '_7dB'
np.random.seed(1)

config = {
    'input_car_size': 4,
    'ant_size': 4,
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
        'is_mse': False,
        'power_distribution_name': 'invariant'
    },
    'path': {
        'ener': 1.0,
        'is_mse': False,
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
    },
    'salt': {
        'ener': 1.0,
        'power_distribution_name': 'invariant'
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
    sum_cos = 0
    for i, input in enumerate(test_loader):
        h_full = input[0].cuda().float()
        # print(in_car, in_ant)
        # print(h_full.shape)
        h_input = h_full[:, :in_ant, :, :]
        h_input = noise_adder.add_noise(h_input)
        h_input = h_input[:, :, in_car, :]
        h_output = h_full[:, :in_ant, out_car, :]
        # print(MSE(h_full[:, :in_ant, in_car, :] - h_input))
        # h_output = h_full[:, out_ant, out_car, :]
        # h_output = h_full
        with torch.no_grad():
            h_now_output = model(h_input)
            nmse = NMSE(h_output, h_now_output)
            sum_nmse += nmse
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss.item()
            sum_loss += loss
            #cos = cosine_similarity(h_output, h_output)
            cos = cosine_similarity(h_output, h_now_output)
            sum_cos += cos
        # print(MSE(h_now_output - h_output), 'output')
        # if i == 10:
        #     exit(1)
    avg_nmse = (sum_nmse / (i + 1))
    test_loss = (sum_loss / (i + 1))
    cos_similarity = (sum_cos / (i + 1))
    return avg_nmse, test_loss, cos_similarity


def test_models(model1, model2, model_name_list, noise_config_list, test_loader, writer1, writer2, in_ant, in_car, out_car):
    for model_name in model_name_list:
        all_nmse = [model_name+'NMSE']
        all_mse = [model_name+'MSE']
        all_cos = [model_name+'cosine_similarity']
        model1 = model1.cuda()
        model1.load_state_dict(torch.load(model_name+'1.pth'))
        model2 = model2.cuda()
        model2.load_state_dict(torch.load(model_name+'2.pth'))
        for noise_config in noise_config_list:
            noise_adder = get_noise_adder(noise_config)
            avg_nmse, test_loss, cos_similarity = get_test_result(model1, test_loader,
                                                                  noise_adder, in_ant, in_car, out_car)
            all_nmse.append(avg_nmse)
            all_mse.append(test_loss)
            all_cos.append(cos_similarity)
        writer1.writerow(all_nmse)
        writer1.writerow(all_mse)
        writer1.writerow(all_cos)
        for noise_config in noise_config_list:
            noise_adder = get_noise_adder(noise_config)
            avg_nmse, test_loss, cos_similarity = get_test_result(model2, test_loader,
                                                                  noise_adder, in_ant, out_car, in_car)
            all_nmse.append(avg_nmse)
            all_mse.append(test_loss)
            all_cos.append(cos_similarity)
        writer2.writerow(all_nmse)
        writer2.writerow(all_mse)
        writer2.writerow(all_cos)


def virtualize_mapping(model_name, model1, model2, ener, num_data):
    model1 = model1.cuda()
    model1.load_state_dict(torch.load(model_name + '1.pth'))
    model2 = model2.cuda()
    model2.load_state_dict(torch.load(model_name + '2.pth'))

    from PIL import Image

    dest_test = "channel_test_32ant_64car_300k.pt"
    channel_test = torch.load(dest_test)
    total_data = channel_test.shape[0]
    torch.manual_seed(1234)
    #samples = torch.tensor(torch.random.choice(channel_test, size=(num_data,), 
    #    dim=0, replace=False))
    perm = torch.randperm(total_data)
    ids = perm[:num_data]
    #print(ids)
    h_full = channel_test[ids, :, :, :]
    h_full = h_full[:, in_ant, :, :].cuda()
    h_input = h_full[:, :, out_car, :]
    h_output = h_full[:, :, in_car, :]

    noise_config = {
        'name': 'Doppler',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'white': {
            'ener': 0.1,
            'is_mse': False,
            'power_distribution_name': 'invariant'
        },
        'path': {
            'ener': 0.1,
            'is_mse': False,
            'antenna_shape': [num_ant],
            'num_path': 2,
            'variant_path': False,
            'variant_power': False
        },
        'Doppler': {
            'Doppler_window_length': 5,
            'alpha': 0.1,
            'alpha_fixed': True,
            'num_car': num_car
        },
        'salt': {
            'ener': 1.0,
            'power_distribution_name': 'invariant'
        }
    }
    noise_adder = get_noise_adder(noise_config)
    h_doppler = noise_adder.add_noise(h_full)
    h_doppler = h_doppler[:, :, out_car, :]
    h_doppler_output = model2(h_doppler)
    NMSE_doppler = NMSE(h_input, h_doppler)
    cos_doppler = cosine_similarity(h_input, h_doppler)
    NMSE_doppler_output = NMSE(h_output, h_doppler_output)
    cos_doppler_output = cosine_similarity(h_output, h_doppler_output)

    noise_config['name'] = 'path'
    noise_adder = get_noise_adder(noise_config)
    h_path = noise_adder.add_noise(h_full)
    h_path = h_path[:, :, out_car, :]
    h_path_output = model2(h_path)
    NMSE_path = NMSE(h_input, h_path)
    cos_path = cosine_similarity(h_input, h_path)
    NMSE_path_output = NMSE(h_output, h_path_output)
    cos_path_output = cosine_similarity(h_output, h_path_output)

    noise_config['name'] = 'white'
    noise_adder = get_noise_adder(noise_config)
    h_white = noise_adder.add_noise(h_full)
    h_white = h_white[:, :, out_car, :]
    h_white_output = model2(h_white)
    NMSE_white = NMSE(h_input, h_white)
    cos_white = cosine_similarity(h_input, h_white)
    NMSE_white_output = NMSE(h_output, h_white_output)
    cos_white_output = cosine_similarity(h_output, h_white_output)

    noise_config['white']['ener'] = ener
    noise_adder = get_noise_adder(noise_config)
    h_ori = noise_adder.add_noise(h_output)
    h_map = model1(h_ori)
    h_map_output = model2(h_map)
    NMSE_map = NMSE(h_input, h_map)
    cos_map = cosine_similarity(h_input, h_map)
    NMSE_map_output = NMSE(h_output, h_map_output)
    cos_map_output = cosine_similarity(h_output, h_map_output)

    def save(i, mat, name):
        mat = mat[i, :, :, 0].detach()
        aaa = torch.max(mat)
        im = Image.fromarray((mat*255/aaa).cpu().numpy().astype('uint8'))
        im.convert('L').save('virtualization/' + str(i) + '/' + name + '.jpg')

    for i in range(num_data):
        os.makedirs('virtualization/' + str(i))
        save(i, h_input, 'groundtruth_in')
        save(i, h_output, 'groundtruth_out')
        save(i, h_doppler, 'doppler_in')
        save(i, h_doppler_output, 'doppler_out')
        save(i, h_path, 'path_in')
        save(i, h_path_output, 'path_out')
        save(i, h_white, 'white_in')
        save(i, h_white_output, 'white_out')
        save(i, h_map, 'map_in')
        save(i, h_map_output, 'map_out')
        with open('virtualization/' + str(i) + '/result.txt', 'w') as f:
            print('doppler input NMSE', NMSE_doppler, 'output NMSE', NMSE_doppler_output,
                  'input cos', cos_doppler, 'output cos', cos_doppler_output, file=f)
            print('path input NMSE', NMSE_path, 'output NMSE', NMSE_path_output,
                  'input cos', cos_path, 'output cos', cos_path_output, file=f)
            print('white input NMSE', NMSE_white, 'output NMSE', NMSE_white_output,
                  'input cos', cos_white, 'output cos', cos_white_output, file=f)
            print('map input NMSE', NMSE_map, 'output NMSE', NMSE_map_output,
                  'input cos', cos_map, 'output cos', cos_map_output, file=f)


model_name1 = 'models/mixer_standard_32ant_64car_300k_'
#model_name2 = ['_7dB_8ant']
model_name2 = ['_7dB_4ant_4car']
model_names = ['white']
model_name_list = []
for n2 in model_name2:
    for n in model_names:
        model_name_list.append(model_name1+n+n2)
noise_config_list = []
# eners = [1, 1.2, 1.4]
# eners = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 10, 15, 20]
eners = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2]
alphas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2]
salt = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
f1 = open('mapping_result_new_noise1_4ant.csv', 'w')
f2 = open('mapping_result_new_noise2_4ant.csv', 'w')

import copy

for e in eners:
    noise_config['white']['ener'] = e
    noise_config_list.append(copy.deepcopy(noise_config))
noise_config['name'] = 'path'
for npath in [1, 2, 3, 4, 5]:
    noise_config['path']['num_path'] = npath
    for e in eners:
        noise_config['path']['ener'] = e
        noise_config_list.append(copy.deepcopy(noise_config))

noise_config['name'] = 'Doppler'
for a in alphas:
    noise_config['Doppler']['alpha'] = a
    noise_config_list.append(copy.deepcopy(noise_config))
noise_config['name'] = 'salt'
for a in salt:
    noise_config['salt']['ener'] = a
    noise_config_list.append(copy.deepcopy(noise_config))

writer1 = csv.writer(f1)
writer1.writerow(eners)
writer1.writerow(alphas)
writer2 = csv.writer(f2)
writer2.writerow(eners)
writer2.writerow(alphas)
test_models(model1, model2, model_name_list, noise_config_list, test_loader, writer1, writer2, config['input_ant_size'], in_car, out_car)

#virtualize_mapping(model_name_list[0], model1, model2, 1.0, 30)

