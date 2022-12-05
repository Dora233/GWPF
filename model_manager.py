import torch, torchvision
from models import CNN, LSTM, ResNet, VGG, AlexNet, DenseNet, LogisticRegression,Transformer
import os, datetime, sys, pprint

CUDA = torch.cuda.is_available()

''' Suggested hyper-parameter values of (learning rate, weight decay, optimizer, loss_func) for each model. '''
ModelHyperparameterDict = { 
        # 'LogisticRegression_Mnist': 
            # {'lr':0.01, 'wd':0, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_Mnist': 
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'CNN_Cifar10':
            # {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'CNN_KWS':
            # {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'SVM_Points':
            # {'lr':0.01, 'wd':0, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'LSTM_KWS':
            # {'lr':0.05, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'ResNet18_Cifar10':
            {'lr':0.01, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'ResNet50_Cifar10':
            # {'lr':0.01, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'VGG11_EMNIST':
            {'lr':0.001, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'VGG16_Cifar10':
            # {'lr':0.001, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'AlexNet_Cifar10':
            # {'lr':0.01, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'ResNet50_Cifar100':
            # {'lr':0.001, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'DenseNet121_Cifar100':
            # {'lr':0.01, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'AlexNet_ImageNet':
            # {'lr':0.1, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'DenseNet121_ImageNet':
            # {'lr':0.01, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'ResNet50_ImageNet':
            # {'lr':0.001, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'ResNet152_ImageNet':
            # {'lr':0.001, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'VGG16_ImageNet':
            # {'lr':0.01, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        # 'Inceptionv3_ImageNet':
            # {'lr':0.01, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'transformer':
            {'lr':0.001, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()}   
    }

class Model_Manager:
    def __init__(self, model_profile):
        self.model_name = model_profile['model_name']
        self.hyperparameters = ModelHyperparameterDict[model_profile['model_name']]

        self.model = None
        self.optimizer = None
        self.loss_func = None

        self.logging('create model manager', self.hyperparameters)  # note that only inner-file hyperparameters (instead of model_profile) is printed

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Model Manager] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    def get_optimizer(self):
        self.optimizer = self.hyperparameters['optimizer'](self.model.parameters(), lr=self.hyperparameters['lr'], weight_decay=self.hyperparameters['wd'])
        return self.optimizer, self.hyperparameters['lr']

    def get_loss_func(self):
        self.loss_func = self.hyperparameters['loss_func']
        return self.loss_func
    
    def load_model(self, CHECKPOINT_ENABLED=True):
        self.load_model_architecture() 
        self.try_init_model_from_checkpoint(CHECKPOINT_ENABLED)
        self.model = self.model.cuda() if CUDA else self.model
        return self.model

    def load_model_architecture(self):
        """ First load model architecture. """
        # if self.model_name == 'LogisticRegression_Mnist':
            # self.model = LogisticRegression.LogisticRegression()
        if self.model_name == 'CNN_Mnist':
            self.model = CNN.CNN_Mnist()
        # if self.model_name == 'CNN_Cifar10':
            # self.model = CNN.CNN_Cifar10()
        # if self.model_name == 'CNN_KWS':
            # self.model = CNN.CNN_KWS()
        # if self.model_name == 'LSTM_KWS':
            # self.model = LSTM.LSTM_KWS()
        if self.model_name == 'ResNet18_Cifar10':
            self.model = ResNet.ResNet18_Cifar10()
        # if self.model_name == 'ResNet50_Cifar10':
            # self.model = torchvision.models.resnet50(num_classes=10)
        if self.model_name == "transformer":
            ntokens = 33278
            emsize = 200
            nhead = 2
            nhid = 200
            nlayers = 2
            dropout = 0.2
            self.model = Transformer.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
        if self.model_name == 'VGG11_EMNIST':
            self.model = VGG.VGG11_EMNIST(num_classes=47)    
            
        # if self.model_name == 'VGG16_Cifar10':
            # self.model = VGG.VGG16_Cifar10()
        # if self.model_name == 'AlexNet_Cifar10':
            # self.model = AlexNet.AlexNet_Cifar10()
        # if self.model_name == 'AlexNet_ImageNet':
            # self.model = AlexNet.AlexNet_ImageNet(num_classes=1000) 
            # self.model = torchvision.models.alexnet(num_classes=1000)
            
        # if self.model_name == 'DenseNet121_Cifar100':
            # self.model = torchvision.models.densenet121(num_classes=100)
        # if self.model_name == 'ResNet50_Cifar100':   
            # self.model = torchvision.models.resnet50(num_classes=100)    
            
        # if self.model_name == 'DenseNet121_ImageNet':
            # self.model = torchvision.models.densenet121(num_classes=1000) 
        #if self.model_name == 'DenseNet201_ImageNet':    
        #    self.model = torchvision.models.densenet201(num_classes=1000)
        # if self.model_name == 'ResNet50_ImageNet':   
            # self.model = torchvision.models.resnet50(num_classes=1000)
        # if self.model_name == 'ResNet152_ImageNet':    
            # self.model = torchvision.models.resnet152(num_classes=1000)
        # if self.model_name == 'VGG16_ImageNet':     
            # self.model = torchvision.models.vgg16(num_classes=1000)
        # if self.model_name == 'Inceptionv3_ImageNet':     
            # self.model = torchvision.models.inception_v3(num_classes=1000)

    def try_init_model_from_checkpoint(self, CHECKPOINT_ENABLED):
        if CHECKPOINT_ENABLED:
            checkpoint_name = self.model_name + '_checkpoint_0.t7'
            if os.path.exists(checkpoint_name):
                self.logging('recover model from: ' + checkpoint_name)
                checkpoint = torch.load(checkpoint_name)
                self.model.load_state_dict(checkpoint['state'])
            else:
                self.save_model(0)
                self.logging('no checkpoint found; randomly initialize model and save it to: ' + checkpoint_name)

    def save_model(self, round_id):
        state = {'state': self.model.state_dict(), 'round': round_id}
        checkpoint_name = self.model_name + '_checkpoint_' + str(round_id) + '.t7'
        if not os.path.exists(checkpoint_name):
            torch.save(state, checkpoint_name)
            self.logging('save model to: ' + checkpoint_name)