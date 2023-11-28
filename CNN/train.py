#coding:utf-8
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from DataRead import read_dataset
# from resnet import ResNet18
from CNN.model.resnet_without_bn import ResNet18

from torch.utils.tensorboard import SummaryWriter

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ������
batch_size = 128
dataset = "cifar10"
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset',dataset = "cifar10")
# ����ģ��(ʹ��Ԥ����ģ�ͣ��޸����һ�㣬�̶�֮ǰ��Ȩ��)
n_class = 10 if dataset == "cifar10" else 100
model = ResNet18()
expansion = 1 
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512 * expansion, n_class) # ������ȫ���Ӳ�ĵ�
model = model.to(device)

# ʹ�ý�������ʧ����
criterion = nn.CrossEntropyLoss().to(device)

# ������¼��
writer = SummaryWriter()


# ��ʼѵ��
n_epochs = 250
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    # ��̬����ѧϰ��
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ###################
    # ѵ������ģ�� #
    ###################
    model.train() #����������batch normalization��drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables������ݶȣ�
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (���򴫵ݣ�ͨ����ģ�ʹ�������������Ԥ�����)
        output = model(data).to(device)  #���ȼ���output = model.forward(data).to(device) ��
        # calculate the batch loss��������ʧֵ��
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # �����򴫵ݣ�������ʧ�����ģ�Ͳ������ݶȣ�
        loss.backward()
        # perform a single optimization step (parameter update)
        # ִ�е����Ż����裨�������£�
        optimizer.step()
        # update training loss��������ʧ��
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # ��֤����ģ��#
    ######################

    model.eval()  # ��֤ģ��
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(���������ת��ΪԤ����)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(��Ԥ������ʵ��ǩ���бȽ�)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy.append(right_sample/total_sample)
 
    # ����ƽ����ʧ
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # ��ʾѵ��������֤������ʧ���� 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    # ��¼ָ��
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', valid_loss, epoch)
    writer.add_scalar('Accuracy/valid', 100*right_sample/total_sample, epoch)

    # �����֤����ʧ�������٣��ͱ���ģ�͡�
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10_without_bn.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1

file = "para/" + "resnet18_cifar10_without_bn" + "_" + "model_parameter_sizes.txt" 

with open(file, "w") as f:
    total_params = 0
    for name, param in model.named_parameters():
        f.write(f"Parameter: {name}, Size: {param.size()}\n")
        total_params += np.prod(param.size())
    f.write(f"Total number of parameters: {total_params}\n")
    print(f"Total number of parameters: {total_params}")

writer.close()