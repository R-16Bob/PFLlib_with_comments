import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# 设置随机种子与目录
random.seed(1)
np.random.seed(1)
num_clients=20
dir_path= "MNIST/"

# 创建目录
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

# Setup directory for train/test data
config_path = dir_path + "config.json"
train_path = dir_path + "train/"
test_path = dir_path + "test/"

# 划分的参数
niid=True
balance=False
partition="dir"

if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    print("check")

# Get MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(
    root=dir_path+"rawdata", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root=dir_path+"rawdata", train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset.data), shuffle=False)

# print(trainloader)
# print(trainset.data)

'''下面做的是：把数据加载器中的数据赋值给trainset和testset，然后再把它们汇集到一个列表中。
然而，实际上我想并不需要赋值这一步'''

for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    testset.data, testset.targets = test_data

# 下面是另一种写法，因为在colab上面的会报错，做的事是一样的。

# for batch_idx, (data, target) in enumerate(trainloader):
#   trainset.data = data
#   trainset.targets = target
#   break
# for batch_idx, (data, target) in enumerate(testloader):
#   testset.data = data
#   testset.targets = target
#   break

dataset_image = []
dataset_label = []

dataset_image.extend(trainset.data.cpu().detach().numpy())
dataset_image.extend(testset.data.cpu().detach().numpy())
dataset_label.extend(trainset.targets.cpu().detach().numpy())
dataset_label.extend(testset.targets.cpu().detach().numpy())
dataset_image = np.array(dataset_image)
dataset_label = np.array(dataset_label)
# print(dataset_image.shape)
# print(dataset_image)

num_classes = len(set(dataset_label))
print(f'Number of classes: {num_classes}')

# print(dataset_label)
# print(dataset_label==0)
# dataset = []
# for i in range(num_classes):
#     idx = dataset_label == i
#     dataset.append(dataset_image[idx])


# 为每个client分配数据
X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                niid, balance, partition, class_per_client=2)
# 划分数据集与测试集
train_data, test_data = split_data(X, y)
# 保存到硬盘
save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
          statistic, niid, balance, partition)

print("注意：为了简便，没有使用函数，所有的参数都是固定的！")
print("因此，check也不会break函数；可以反复运行查看结果。")