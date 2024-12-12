import os
import random
from model.model import *

# from aptflops import get_model_complexity_info

#GPU调用
gpu_list = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

np.random.seed(1)
random.seed(1)

#超参数定义
batch_size=256
epochs=200
learning_rate=1e-3
print_freq=200
input_car_size=1 #在mapping中输入的载波数量
ant_size=1 #总天线数量
input_ant_size=1 #在mapping中输入的天线数量
car_size=64-input_car_size #总载波数量

noise_var = 1.0
sqrt_var = math.sqrt(noise_var)
is_different_var = True
depth=5#使用的CMLP-Mixer Layer的深度


# 读取训练集和测试集
path = '/mnt/HD2/yyz/MIMOlocdata/'
name1 = '8ant_64car_300k'
name2 = '_input%d_noise%f_' % (input_car_size, noise_var)
if is_different_var:
    name = name1 + name2 + 'variant'
else:
    name = name1 + name2 + 'invariant'

freqs = []
for i in range(input_car_size):
    freqs.append(2+i*60/input_car_size)
datasets = get_dataset([0.8, 0.2], freqs=freqs, dataset_path=path, name='32ant_64car_300k')
train_loader = torch.utils.data.DataLoader(
    datasets[0], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets[1], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

#定义模型，读取计算量和参数量
model=Mapping_Net(input_ant_size,input_car_size,ant_size,car_size,depth)
# macs, params = get_model_complexity_info(model, (input_ant_size,input_car_size,2), print_per_layer_stat=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# print('#model parameters:', sum(param.numel() for param in model.parameters()))

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def add_noise(input):
    batch_size = input.shape[0]
    noise = torch.randn_like(input)
    if is_different_var:
        v = torch.rand(batch_size, 1, 1, 1).cuda() * sqrt_var
        return input + noise * v
    else:
        return input + noise * sqrt_var


#调整学习率
def adjust_rate(epoch,optim,lr):
    if epoch < 100:
        optim.param_groups[0]['lr'] = lr * (0.2 ** 0)
        return
    elif epoch >= 100 and epoch < 150:
        optim.param_groups[0]['lr'] = lr * (0.2 ** 1)
        return
    elif epoch>=150 and epoch<200:
        optim.param_groups[0]['lr'] = lr * (0.2 ** 2)
        return
import csv
f = open('models/pretrain1'+name+'.csv', 'w')
writer = csv.writer(f)
writer.writerow(['epoch','training_loss','training_NMSE','testing_loss','testing_NMSE'])
#训练和测试
nmse_list=[]
for epoch in range(epochs):
    model.train()
    adjust_rate(epoch,optimizer,learning_rate)
    # print('lr:%.4e' % optimizer.param_groups[0]['lr'])

    sum_nmse = 0
    sum_loss = 0
    for i, input in enumerate(train_loader):
        optimizer.zero_grad()

        h_input = input[0].cuda().float()
        h_input = h_input[:, :ant_size, :, :]
        h_input = add_noise(h_input)
        h_output = input[1].cuda().float()
        h_output = h_output[:, :ant_size, :, :]
        h_now_output = model(h_input)
        MSE_loss = nn.MSELoss()(h_now_output, h_output)
        loss = MSE_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            nmse = Nmse(h_output, h_now_output)
            sum_nmse += nmse
            sum_loss += loss.item()
        # if epoch % 20 == 0:
        #     if i % print_freq==0:
        #         print("i", i)
        #         print("epoch :", epoch)
        #         print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        #         print("MSELoss : ", loss.item())
        #         print("nmse : ", nmse)
    train_avg_nmse = (sum_nmse / (i + 1))
    train_loss = sum_loss / (i + 1)

    # 每20个epoch做一次测试，测试结果是所有测试集样本的平均结果
    # if epoch % 20 == 0:
    print("test")
    # torch.save(model.state_dict(),'./model' + '.pth')
    model.eval()
    sum_nmse=0
    sum_loss=0
    for i, input in enumerate(test_loader):
        h_input = input[0].cuda().float()
        h_input = h_input[:, :ant_size, :, :]
        h_input = add_noise(h_input)
        h_output = input[1].cuda().float()
        h_output = h_output[:, :ant_size, :, :]
        with torch.no_grad():
            h_now_output = model(h_input)
            nmse = Nmse(h_output, h_now_output)
            sum_nmse+=nmse
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss.item()
            sum_loss += loss
    avg_nmse=(sum_nmse/(i+1))
    test_loss = (sum_loss / (i + 1))
    print("avg_nmse : ", avg_nmse)
    # nmse_list.append(avg_nmse)
    # nmse_array = np.array(nmse_list)
    # np.save("nmse.npy", nmse_array)
    writer.writerow([epoch, train_loss, train_avg_nmse, test_loss, avg_nmse])

# #最终的测试
# model.eval()
# sum_nmse=0
# print("final test")
# for i, input in enumerate(test_loader):
#     h_input = input[0].cuda().float()
#     h_output = input[1].cuda().float()
#     with torch.no_grad():
#         h_now_output = model(h_input)
#         nmse = Nmse(h_output, h_now_output)
#         sum_nmse+=nmse
# avg_nmse=(sum_nmse/(i+1))
# print("avg_nmse : ", avg_nmse)
torch.save(model.state_dict(),'./models/model1'+name+'.pth')



#定义模型，读取计算量和参数量
model=Mapping_Net(input_ant_size,car_size,ant_size,input_car_size,depth)
# macs, params = get_model_complexity_info(model, (input_ant_size,car_size,2), print_per_layer_stat=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# print('#model parameters:', sum(param.numel() for param in model.parameters()))

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
f = open('models/pretrain2'+name+'.csv', 'w')
writer = csv.writer(f)
writer.writerow(['epoch','training_loss','training_NMSE','testing_loss','testing_NMSE'])
#训练和测试
nmse_list=[]
for epoch in range(epochs):
    model.train()
    adjust_rate(epoch,optimizer,learning_rate)
    # print('lr:%.4e' % optimizer.param_groups[0]['lr'])

    sum_nmse = 0
    sum_loss = 0
    for i, input in enumerate(train_loader):
        optimizer.zero_grad()

        h_input = input[1].cuda().float()
        h_input = h_input[:, :ant_size, :, :]
        h_input = add_noise(h_input)
        h_output = input[0].cuda().float()
        h_output = h_output[:, :ant_size, :, :]
        h_now_output = model(h_input)
        MSE_loss = nn.MSELoss()(h_now_output, h_output)
        loss = MSE_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            nmse = Nmse(h_output, h_now_output)
            sum_nmse += nmse
            sum_loss += loss.item()
        # if epoch % 20 == 0:
        #     if i % print_freq==0:
        #         print("i", i)
        #         print("epoch :", epoch)
        #         print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        #         print("MSELoss : ", loss.item())
        #         print("nmse : ", nmse)
    train_avg_nmse = (sum_nmse / (i + 1))
    train_loss = sum_loss / (i + 1)

    # 每20个epoch做一次测试，测试结果是所有测试集样本的平均结果
    # if epoch % 20 == 0:
    print("test")
    # torch.save(model.state_dict(),'./model' + '.pth')
    model.eval()
    sum_nmse=0
    sum_loss=0
    for i, input in enumerate(test_loader):
        h_input = input[1].cuda().float()
        h_input = h_input[:, :ant_size, :, :]
        h_input = add_noise(h_input)
        h_output = input[0].cuda().float()
        h_output = h_output[:, :ant_size, :, :]
        with torch.no_grad():
            h_now_output = model(h_input)
            nmse = Nmse(h_output, h_now_output)
            sum_nmse+=nmse
            MSE_loss = nn.MSELoss()(h_now_output, h_output)
            loss = MSE_loss.item()
            sum_loss += loss
    avg_nmse=(sum_nmse/(i+1))
    test_loss = (sum_loss / (i + 1))
    print("avg_nmse : ", avg_nmse)
    # nmse_list.append(avg_nmse)
    # nmse_array = np.array(nmse_list)
    # np.save("nmse.npy", nmse_array)
    writer.writerow([epoch, train_loss, train_avg_nmse, test_loss, avg_nmse])
torch.save(model.state_dict(),'./models/model2'+name+'.pth')
