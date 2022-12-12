import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

data_path = '/YOURdataPATH/data'
data_path_wiki = '/YOURdataPATH/data/wikitext-2'
#---------------------------------------
DIRICHLET_ALPHA = 1
#---------------------------------------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, seed):
        self.dataset = dataset
        self.idxs = list(idxs)
        unique, counts = np.unique(np.array(self.dataset.targets)[self.idxs], return_counts=True)
        random.seed(seed)
        random.shuffle(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def dirichlet_split_noniid(train_labels, alpha, n_clients, seed):
    n_classes = train_labels.max()+1
    np.random.seed(seed)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs2 = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs2

class Dataset_Manager:
    def __init__(self, dataset_profile): 
        self.dataset_name = dataset_profile['dataset_name']
        self.is_iid = dataset_profile['is_iid']
        self.total_partition_number = dataset_profile['total_partition_number']
        self.partition_rank = dataset_profile['partition_rank']

        self.batch_size = 100 if dataset_profile['dataset_name'] != 'ImageNet' else 128
        print('[Dataset Manager]')
        self.train_sampler = None
        
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
            dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
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
            dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
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

            dict_users = dirichlet_split_noniid(np.array(train_set.targets), alpha=DIRICHLET_ALPHA, n_clients=self.total_partition_number, seed=seed)
        elif self.dataset_name == "wikitext2":
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


def wikitext_iid(dataset, num_users):
    data_len = len(dataset)
    dict_users = []
    sizes = [1.0 / num_users for _ in range(num_users)]
    indexes = [x for x in range(0, data_len)]
    for frac in sizes:
        part_len = int(frac * data_len)
        dict_users.append(dataset[indexes[0: part_len]])
    return dict_users

def wikitext_noniid(dataset, num_users, seed):
    data_len = len(dataset)
    dict_users = []
    sizes = gen_partition(data_len, num_users, seed)
    indexes = [x for x in range(0, data_len)]
    for part_len in sizes:
        dict_users.append(dataset[indexes[0: part_len]])
    return dict_users


def gen_partition(_sum, num_users, seed):
    _seed = seed
    mean = _sum / num_users
    variance = _sum / 2

    min_v = 1
    max_v = mean + variance
    array = [min_v] * num_users

    diff = _sum - min_v * num_users
    while diff > 0:
        np.random.seed(_seed)
        a = np.random.randint(0, num_users)
        if array[a] >= max_v:
            continue
        np.random.seed(_seed)
        delta = np.random.randint(1, diff // 4 + 4)
        array[a] += delta
        diff -= delta
        array[a] += diff if diff < 0 else 0
        _seed += 2
    return np.array(array)


def iid_part(dataset, num_users, seed):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # Random choose num_items images from all images.
        np.random.seed(seed)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_part(dataset, num_users, num_shards, num_imgs, seed):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # sort the label by the target value
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    idx_shard_partition = gen_partition(num_shards, num_users, seed)
    for i in range(num_users):
        np.random.seed(seed)
        rand_set = set(np.random.choice(idx_shard, int(idx_shard_partition[i]), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
