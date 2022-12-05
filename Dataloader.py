import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from Utils.Sample import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid, cifar100_iid,\
    cifar100_noniid, emnist_iid, emnist_noniid, wikitext_iid, wikitext_noniid

data_path = '/home/zjlab/yangduo/Overlap-FedAvg-main/data'
data_path_wiki = '/home/zjlab/yangduo/Overlap-FedAvg-main/data/wikitext-2'
#---------------------------------------
DIRICHLET_ALPHA = 1
#---------------------------------------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, seed):
        self.dataset = dataset
        self.idxs = list(idxs)
        unique, counts = np.unique(np.array(self.dataset.targets)[self.idxs], return_counts=True)
        #print(dict(zip(unique, counts)))
        random.seed(seed)
        random.shuffle(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def dirichlet_split_noniid(train_labels, alpha, n_clients, seed):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    np.random.seed(seed)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    # worker = [[]]*n_clients
    # for i in range(n_clients):
    #     worker[i] = [len(a) for a in client_idcs[i]]
    # worker2 = np.array(worker)
    # np.savetxt('data.txt',worker2)
    client_idcs2 = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs2

class Dataset_Manager:
    def __init__(self, dataset_profile): # dataset_name, is_iid, total_partition_number, rank):
        self.dataset_name = dataset_profile['dataset_name']
        self.is_iid = dataset_profile['is_iid']
        self.total_partition_number = dataset_profile['total_partition_number']
        self.partition_rank = dataset_profile['partition_rank']

        self.batch_size = 100 if dataset_profile['dataset_name'] != 'ImageNet' else 128
        #self.logging('create dataset') # no special hyperparameter here for different dataset types
        print('[Dataset Manager]')
        self.train_sampler = None

    # def logging(self, string, hyperparameters=None):
        # print('['+str(datetime.datetime.now())+'] [Dataset Manager] '+str(string))
        # if hyperparameters != None:
            # pprint.pprint(hyperparameters)
        # sys.stdout.flush()
        
    def partition_dataset(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        if self.dataset_name == "mnist":
            train_set = datasets.MNIST(data_path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            test_set = datasets.MNIST(data_path, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
            if self.is_iid:
                dict_users = mnist_iid(train_set, self.total_partition_number, seed)
            else:
                #dict_users = mnist_noniid(train_set, self.total_partition_number, seed)
                dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
        elif self.dataset_name == "fmnist":
            train_set = datasets.FashionMNIST(data_path, train=True, download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))
            test_set = datasets.FashionMNIST(data_path, train=False, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
            if self.is_iid:
                dict_users = mnist_iid(train_set, self.total_partition_number, seed)
            else:
                dict_users = mnist_noniid(train_set, self.total_partition_number, seed)
        elif self.dataset_name == "cifar10":
            train_set = datasets.CIFAR10(data_path, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ]))
            test_set = datasets.CIFAR10(data_path, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
            if self.is_iid:
                dict_users = cifar10_iid(train_set, self.total_partition_number, seed)
            else:
                #dict_users = cifar10_noniid(train_set, self.total_partition_number, seed)
                dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
        elif self.dataset_name == "cifar100":
            train_set = datasets.CIFAR100(data_path, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                          ]))
            test_set = datasets.CIFAR100(data_path, train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                         ]))
            if self.is_iid:
                dict_users = cifar100_iid(train_set, self.total_partition_number, seed)
            else:
                dict_users = cifar100_noniid(train_set, self.total_partition_number, seed)
        elif self.dataset_name == "emnist":
            train_set = datasets.EMNIST(data_path, train=True, download=True, split="balanced",
                                        transform=transforms.Compose([
                                            transforms.Resize(size=32),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
            test_set = datasets.EMNIST(data_path, train=False, split="balanced",
                                transform=transforms.Compose([
                                    transforms.Resize(size=32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))

            if self.is_iid:
                dict_users = emnist_iid(train_set, self.total_partition_number, seed)
            else:
                #dict_users = emnist_noniid(train_set, self.total_partition_number, seed)
                dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
        elif self.dataset_name == "wikitext2":
            #corpus = Corpus("rnn_data/wikitext-2") 
            corpus = Corpus(data_path_wiki)
            train_set, test_set = corpus.train, corpus.test

        if self.dataset_name == "wikitext2":
            if self.is_iid:
                dict_users = wikitext_iid(train_set, self.total_partition_number)
            else:
                dict_users = wikitext_noniid(train_set, self.total_partition_number, seed)
            train_loader = batchify(dict_users[self.partition_rank], self.batch_size)
            eval_batch_size = 10
            val_loader = batchify(test_set, eval_batch_size)
        else:
            #print(self.dataset_name, dict_users)
            train_loader = DataLoader(DatasetSplit(train_set, dict_users[self.partition_rank], seed), batch_size=self.batch_size, shuffle=False)
            val_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
            print(next(iter(train_loader))[1])


        return train_loader, val_loader


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // int(bsz)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * int(bsz))
    # Evenly divide the data across the bsz batches.
    data = data.view(int(bsz), -1).t().contiguous()
    return data