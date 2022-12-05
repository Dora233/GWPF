# -*- coding: utf-8 -*-
from __future__ import print_function
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random, datetime, sys, pprint

import os
#from datasets import DatasetHDF5

class DatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, hdf5fn, t, transform=None, target_transform=None):
        """
        t: 'train' or 'val'
        """
        super(DatasetHDF5, self).__init__()
        self.hf = h5py.File(hdf5fn, 'r', libver='latest', swmr=True)
        self.t = t
        self.n_images= self.hf['%s_img'%self.t].shape[0]
        self.dlabel = self.hf['%s_labels'%self.t][...]
        self.d = self.hf['%s_img'%self.t]
        self.transform = transform
        self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.d[index, ...]
        target = self.dlabel[index]
        return img, np.int64(target)

    def __getitem__(self, index):
        img, target = self._get_dataset_x_and_target(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.n_images

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append((info[0], int(info[1])))
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)


class Dataset_Manager: 
    def __init__(self, dataset_profile): # dataset_name, is_iid, total_partition_number, rank):
        self.dataset_name = dataset_profile['dataset_name']
        self.is_iid = dataset_profile['is_iid']
        self.total_partition_number = dataset_profile['total_partition_number']
        self.partition_rank = dataset_profile['partition_rank']

        self.batch_size = 100 if dataset_profile['dataset_name'] != 'ImageNet' else 128
        self.training_dataset = self.get_training_dataset()
        self.testing_dataset = self.get_testing_dataset()

        self.logging('create dataset') # no special hyperparameter here for different dataset types
        
        self.train_sampler = None

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Dataset Manager] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def get_training_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset_name == 'Cifar100':
            dataset = datasets.CIFAR100(root='./Datasets/cifar100/', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./Datasets/kws/index_train.txt', transform=None)
        if self.dataset_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            hdf5fn = os.path.join('/data/backup_data/imagenet2012_hdf5/', 'imagenet-shuffled-224.hdf5')
            dataset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([transforms.ToPILImage(),
                transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize,]))
            # dataset = datasets.ImageFolder('/data/backup_data/imagenet2012/train', transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # normalize,]))
        return dataset
        
    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_size = 224
        #image_size = 128
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 1000)

        # hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
        hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled-224.hdf5')
        #hdf5fn = os.path.join(self.data_dir, 'imagenet-2012.hdf5')
        #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
        self.trainset = trainset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)
        #testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
            
    def get_testing_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/mnist/', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/cifar10/', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar100':
            dataset = datasets.CIFAR100(root='./Datasets/cifar100/',  train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./Datasets/kws/index_test.txt', transform=None)
        if self.dataset_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            hdf5fn = os.path.join('/data/backup_data/imagenet2012_hdf5/', 'imagenet-shuffled-224.hdf5')
            dataset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([transforms.ToPILImage(),
                    transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ]))
            # dataset = datasets.ImageFolder('/data/backup_data/imagenet2012/val', transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # normalize,]))
        return dataset

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_index_partition(self):
        labels = np.array(self.training_dataset.targets)
        idxs = np.argsort(labels) #sort elements from small to big and output their index
        each_part_size = len(self.training_dataset) // self.total_partition_number
        return idxs[each_part_size*self.partition_rank : each_part_size*(self.partition_rank+1)]
        
    def get_training_dataloader(self):
        
        #if self.is_iid:
        #    self.set_seed(self.partition_rank)
        #    training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=shuffle_, num_workers=self.total_partition_number, pin_memory=True)
        #else:
        #    self.set_seed(0)
        #    index_partition = self.get_index_partition()
        #    training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition), batch_size=self.batch_size, shuffle=False)
        self.train_sampler = None
        shuffle_ = True
        if self.total_partition_number > 1: 
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.training_dataset, num_replicas=self.total_partition_number, rank=self.partition_rank)
            self.train_sampler.set_epoch(0)
            shuffle_ = False
        training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=shuffle_, num_workers=1, pin_memory=True, sampler=self.train_sampler)
        
        return training_dataloader

    def get_testing_dataloader(self):
        testing_dataloader = DataLoader(dataset=self.testing_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        return testing_dataloader
