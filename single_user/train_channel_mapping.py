# Pretraining channel mapping NN
import torch
import torch.nn as nn
import numpy as np
import os
import json
import csv
from model.model import get_mixer_net
from noise import get_noise_adder
from utils import get_dataset, NMSE


def train(config):
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

        def adjust_rate(epoch, optim, lr):
            if epoch < 100:
                optim.param_groups[0]['lr'] = lr * (0.2 ** 0)
                return
            elif 100 <= epoch < 150:
                optim.param_groups[0]['lr'] = lr * (0.2 ** 1)
                return
            elif 150 <= epoch < 200:
                optim.param_groups[0]['lr'] = lr * (0.2 ** 2)
                return
    else:
        raise NotImplementedError
    print(model1, model2)

    noise_adder = get_noise_adder(config['noise'])
    test_noise_adder = get_noise_adder(config['noise_test'])

    # Dataset
    datasets = get_dataset(ratio=config['ratio'], seed=1234,
                           dataset_path=config['dataset_path'], name=config['dataset_name'])
    train_loader = torch.utils.data.DataLoader(
        datasets[0], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets[1], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model1 = model1.cuda()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    model2 = model2.cuda()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    f1 = open('results/' + experiment_name + '1.csv', 'w')
    writer1 = csv.writer(f1)
    writer1.writerow(['epoch', 'training_loss', 'training_NMSE', 'testing_loss', 'testing_NMSE', 'testing_loss_noisy', 'testing_NMSE_noisy'])
    f2 = open('results/' + experiment_name + '2.csv', 'w')
    writer2 = csv.writer(f2)
    writer2.writerow(['epoch', 'training_loss', 'training_NMSE', 'testing_loss', 'testing_NMSE', 'testing_loss_noisy', 'testing_NMSE_noisy'])

    in_car = []
    out_car = []
    for i in range(0, config['car_size'], (config['car_size']-1)//(config['input_car_size']-1)):
        in_car.append(i)
    for i in range(config['car_size']):
        if i not in in_car:
            out_car.append(i)
    print(in_car, out_car)

    print("Model 1 train start!")
    for epoch in range(epochs):
        model1.train()
        adjust_rate(epoch, optimizer1, learning_rate)
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        sum_nmse = 0
        sum_loss = 0
        for i, input in enumerate(train_loader):
            optimizer1.zero_grad()

            h_full = input[0].cuda().float()
            h_input = h_full[:, :config['input_ant_size'], in_car, :]
            h_input = noise_adder.add_noise(h_input)
            h_output = h_full[:, :config['input_ant_size'], out_car, :]
            # h_output = h_full
            h_now_output = model1(h_input)
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss
            loss.backward()
            optimizer1.step()
            with torch.no_grad():
                nmse = NMSE(h_output, h_now_output)
                sum_nmse += nmse
                sum_loss += loss.item()
        train_avg_nmse = (sum_nmse / (i + 1))
        train_loss = sum_loss / (i + 1)

        # torch.save(model.state_dict(),'./model' + '.pth')
        model1.eval()
        sum_nmse = 0
        sum_loss = 0
        sum_nmse_noisy = 0
        sum_loss_noisy = 0
        for i, input in enumerate(test_loader):
            h_full = input[0].cuda().float()
            h_input = h_full[:, :config['input_ant_size'], in_car, :]
            h_input_noisy = test_noise_adder.add_noise(h_input)
            h_output = h_full[:, :config['input_ant_size'], out_car, :]
            # h_output = h_full[:, out_ant, out_car, :]
            # h_output = h_full
            with torch.no_grad():
                h_now_output = model1(h_input)
                nmse = NMSE(h_output, h_now_output)
                sum_nmse += nmse
                MSE_loss = nn.MSELoss()(h_now_output, h_output)
                loss = MSE_loss.item()
                sum_loss += loss

                h_now_output = model1(h_input_noisy)
                nmse = NMSE(h_output, h_now_output)
                sum_nmse_noisy += nmse
                MSE_loss = nn.MSELoss()(h_now_output, h_output)
                loss = MSE_loss.item()
                sum_loss_noisy += loss
        avg_nmse = (sum_nmse / (i + 1))
        test_loss = (sum_loss / (i + 1))
        avg_nmse_noisy = (sum_nmse_noisy / (i + 1))
        test_loss_noisy = (sum_loss_noisy / (i + 1))
        print("avg_nmse : ", avg_nmse)
        writer1.writerow([epoch, train_loss, train_avg_nmse, test_loss, avg_nmse, test_loss_noisy, avg_nmse_noisy])

    # torch.save(model1.state_dict(), 'models/'+experiment_name+'1.pth')\

    print("Model 2 train start!")
    for epoch in range(epochs):
        model2.train()
        adjust_rate(epoch, optimizer2, learning_rate)
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        sum_nmse = 0
        sum_loss = 0
        for i, input in enumerate(train_loader):
            optimizer2.zero_grad()

            h_full = input[0].cuda().float()
            h_input = h_full[:, :config['input_ant_size'], out_car, :]
            h_input = noise_adder.add_noise(h_input)
            h_output = h_full[:, :config['input_ant_size'], in_car, :]
            # h_output = h_full
            h_now_output = model2(h_input)
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss
            loss.backward()
            optimizer2.step()
            with torch.no_grad():
                nmse = NMSE(h_output, h_now_output)
                sum_nmse += nmse
                sum_loss += loss.item()
        train_avg_nmse = (sum_nmse / (i + 1))
        train_loss = sum_loss / (i + 1)

        model2.eval()
        sum_nmse = 0
        sum_loss = 0
        sum_nmse_noisy = 0
        sum_loss_noisy = 0
        for i, input in enumerate(test_loader):
            h_full = input[0].cuda().float()
            h_input = h_full[:, :config['input_ant_size'], out_car, :]
            h_input_noisy = test_noise_adder.add_noise(h_input)
            h_output = h_full[:, :config['input_ant_size'], in_car, :]
            # h_output = h_full[:, out_ant, out_car, :]
            #h_output = h_full
            with torch.no_grad():
                h_now_output = model2(h_input)
                nmse = NMSE(h_output, h_now_output)
                sum_nmse += nmse
                MSE_loss = nn.MSELoss()(h_now_output, h_output)
                loss = MSE_loss.item()
                sum_loss += loss

                h_now_output = model2(h_input_noisy)
                nmse = NMSE(h_output, h_now_output)
                sum_nmse_noisy += nmse
                MSE_loss = nn.MSELoss()(h_now_output, h_output)
                loss = MSE_loss.item()
                sum_loss_noisy += loss
        avg_nmse = (sum_nmse / (i + 1))
        test_loss = (sum_loss / (i + 1))
        avg_nmse_noisy = (sum_nmse_noisy / (i + 1))
        test_loss_noisy = (sum_loss_noisy / (i + 1))
        print("avg_nmse : ", avg_nmse)
        writer2.writerow([epoch, train_loss, train_avg_nmse, test_loss, avg_nmse, test_loss_noisy, avg_nmse_noisy])
    # torch.save(model2.state_dict(), 'models/' + experiment_name + '2.pth')


gpu_list = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
experiment_suffix = '_noiseless_8ant_new'
np.random.seed(1)


for name in ['white']:
    config = {
        'input_car_size': 4,
        'ant_size': 32,
        'input_ant_size': 8,
        'car_size': 64,

        'dataset_path': '/mnt/HD2/yyz/MIMOnoisedata/',
        'dataset_name': '32ant_64car_300k',
        'ratio': [0.8, 0.2],
        'model_name': 'mixer_standard'
    }
    num_car = config['car_size']
    num_ant = config['input_ant_size']
    noise_config = {
        'name': name,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'white': {
            'ener': 0,
            'is_mse': True,
            'power_distribution_name': 'uniform'
        },
        'path': {
            'ener': 1,
            'is_mse': True,
            'antenna_shape': [num_ant],
            'num_path': 3,
            'variant_path': True,
            'variant_power': True
        },
        'Doppler': {
            'Doppler_window_length': 5,
            'alpha': 0.1,
            'alpha_fixed': False,
            'num_car': num_car
        }
    }
    noise_config_test = {
        'name': name,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'white': {
            'ener': 1,
            'is_mse': True,
            'power_distribution_name': 'invariant'
        },
        'path': {
            'ener': 1,
            'is_mse': True,
            'antenna_shape': [num_ant],
            'num_path': 3,
            'variant_path': True,
            'variant_power': True
        },
        'Doppler': {
            'Doppler_window_length': 5,
            'alpha': 0.1,
            'alpha_fixed': False,
            'num_car': num_car
        }
    }
    config['noise'] = noise_config
    config['noise_test'] = noise_config_test
    experiment_name = config['model_name'] + '_' + config['dataset_name'] \
                      + '_' + config['noise']['name'] + experiment_suffix
    print(config)
    print('Experiment', experiment_name, 'start!')
    with open('configs/' + experiment_name + '.json', 'w') as f:
        json.dump(config, f)
    # with open(name, 'r') as f:
    #     config = json.load(f)
    train(config)
