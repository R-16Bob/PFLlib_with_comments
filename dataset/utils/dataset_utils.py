# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1 # for Dirichlet distribution

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))
    # 左：保证至少有一个batch的测试数据；
    dataidx_map = {}

    if not niid:  # IID实际上就是每个client拥有所有类的pat
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        # print(idxs)
        idx_for_each_class = []
        # 通过布尔索引选出每个类对应的idx列表
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])
        # print(idx_for_each_class)


        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):  # 处理每一个类
            selected_clients = []
            for client in range(num_clients):  # 统计哪些client拥有该类
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]
            print('selected_clients: ',selected_clients)
            # 采样率q=(class_per_client/num_classes),按采样率截取client;对IID来说采样率为1，也就是全部client;
            # 对pat，若10client选2类，就是0.2采样率，有10*0.2=2个client有该类样本；若20cleint选2类，就是0.2*20=4个client有该类样本
            num_all_samples = len(idx_for_each_class[i])  # i类样本数量
            num_selected_clients = len(selected_clients)  # 该类被选中的client数量
            num_per = num_all_samples / num_selected_clients  # 该类每个client拥有的样本数量
            if balance:  # balance的话，所有clients平分样本
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]  # 这里少分了一个client，因为其实已经分了一个
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
                # 用randint生成随机列表，low=max(num_per/10, least_samples/num_classes),high=num_per,size就是少分一个
                # 我理解的话，就是除了最后一个，其他所有的样本都是最小到平均值之间的，最后那个client是剩余的所有样本。结果就是一个很大的和其他很小的。
            num_samples.append(num_all_samples-sum(num_samples))  # 剩下的一个client被分配剩余的样本，目的是不能均分时的处理

            # 上面确定了每个client的样本数量，现在分配给每个client
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    # 实际上是截取了布尔索引的一部分
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:  # 如果样本最小值不满足要求，就重新分配
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]  # [0]取了索引值，因为where的返回值是tuple，所以要取0
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # print('prop: ',proportions)
                # 解释：python中True和False可以被当作1和0进行数学运算;意思是将len(idx_j)<N/num_clients不成立的p赋0
                # 也就是idx_j的长度大于等于N/num_clients，则p赋0；目的是，当某个client的样本数量达到平均值时，就不再给它分配样本了；
                # 也就是设定了每个client的样本上限就是平均分配的值。
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                # print('new prop:',proportions)
                proportions = proportions/proportions.sum()  # 对狄利克雷分布进行归一化
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]  # cumsum为累积求和，对prop进行调整
                # split 将idx_k按照proportions进行分割；然后在idx_batch中追加新的索引
                print(len(idx_batch),len(idx_batch[0]))
                # 也就是说，将第k类的索引按照狄利克雷分布（也就是20个client分别分到的数据）的比例进行划分；
                # 随后，将这个索引添加到对应client的idx_batch中；达到的效果就是，对每个类根据狄利克雷分布进行分配。
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]

                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data 根据索引map为每个client分配实际数据
    for client in range(num_clients):
        idxs = dataidx_map[client]
        # print('idxs: ',idxs)
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            # 字典列表存放的是元组，内容是每个标签和该标签的样本数
            
    # 垃圾回收释放内存
    del data
    # gc.collect()

    # 基于X和y打印每个client的信息
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    '''
    X: 记录每个client的数据内容的下标
    y: 记录每个client拥有的标签的下标
    statistic: 记录每个client拥有的数据类型及数量
    '''
    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}
    # 遍历每个client
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)  # 明确键名data保存数据，读取时需要['data']
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
