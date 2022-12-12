import torch
import sys, os, argparse, copy, numpy, pprint, datetime,time,math
from model_manager import Model_Manager
from Dataloader import Dataset_Manager
from sync_manager import Sync_Manager
import torch.nn.functional as F
CUDA = torch.cuda.is_available()

class Client:
    def __init__(self, local_profile, sync_profile, trial_no, dataset_model, is_iid):
        self.model_manager = Model_Manager(local_profile['model_profile'])
        self.model_name = local_profile['model_profile']['model_name']
        self.model = self.model_manager.load_model()
        self.optimizer, self.lr = self.model_manager.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[5,10],gamma = 0.1)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
        self.training_dataloader, self.testing_dataloader = self.dataset_manager.partition_dataset(seed=3000)
        self.max_epoch = 100 

        self.sync_manager = Sync_Manager(self.model, sync_profile)
        self.rank = sync_profile['dist_profile']['rank']
        
        self.savenum = trial_no
        self.dataset_model = dataset_model
        if self.dataset_model == 'B1' or 'B2':
            self.max_epoch = 10000
        
        self.is_iid = is_iid
        self.f_acc = open(os.path.join('Latest_Log/','client0_test_acc.txt'), 'w')

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client] '+str(string))
        sys.stdout.flush()
    
    def test(self):
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        loss_out_put = 0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(self.testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = self.model(test_x)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y) 
        accuracy = positive_test_number / total_test_number
        return accuracy

    def get_batch(self, source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def get_accuracy(self, rnn):
        self.model.eval()
        correct_sum = 0.0
        val_loss = 0.0
        with torch.no_grad():
            if rnn:
                bptt = 35
                ntokens = 33278
                num_batches = 0
                for i in range(0, self.testing_dataloader.size(0) - 1, bptt):
                    num_batches += 1
                    inputs, target = self.get_batch(self.testing_dataloader, i, bptt)
                    if CUDA:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    output = self.model(inputs)
                    output = output.view(-1, ntokens)
                    val_loss += len(inputs) * F.nll_loss(output, target).item()

                val_loss /= (len(self.testing_dataloader) - 1)
                return math.exp(val_loss)
            else:
                for i, (data, target) in enumerate(self.testing_dataloader):
                    if CUDA:
                        data = data.cuda()
                        target = target.cuda()
                    out = self.model(data)
                    pred = out.argmax(dim=1, keepdim=True)
                    if CUDA:
                        pred = pred.cuda()
                        target = target.cuda()
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    correct_sum += correct

                acc = correct_sum / len(self.testing_dataloader.dataset)
                return acc

    def train(self):
        print('\n\n\t-------- START TRAINING --------\n')
        iter_id, epoch_id = 0, 0
        ratio = 0
        time0 = time.time()
        print('time0',time0)
        while epoch_id < self.max_epoch:
            epoch_id += 1 
            self.logging('start epoch: %d; lr: %f' % (epoch_id, self.lr))
            total_loss = 0.
            if self.dataset_model == "W1":
                bptt = 35
                ntokens = 33278
                for batch, i in enumerate(range(0, self.training_dataloader.size(0) - 1, bptt)):
                    train_start = time.time()
                    iter_id += 1
                    data, target = self.get_batch(self.training_dataloader, i, bptt)
                    if CUDA:
                        data = data.cuda()
                        target = target.cuda()
                    self.model.zero_grad()
                    pred = self.model(data)
                    loss = self.loss_func(pred.view(-1, ntokens), target)
                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    train_end = time.time()
                    
                    True_Flase, ratio = self.sync_manager.try_sync_model(iter_id)
                    sync_end = time.time()
                    if True_Flase: 
                        if self.rank == 0:
                            rnn = (self.model_name == "transformer")
                            accuracy = self.get_accuracy(rnn = rnn)
                            self.scheduler.step(accuracy)
                            self.logging(' - test - iter_id: %d; epoch_id: %d; round_id: %d; accuracy: %.4f; loss: %.4f; frozen_ratio: %.4f;' % (iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy, total_loss/len(self.training_dataloader), ratio))
                            self.f_acc.write('%.4f, %d, %d, %d, %.4f, %.4f, %.4f, %.4f, %.4f\n' % (time.time()-time0, iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy, total_loss/len(self.training_dataloader), ratio, \
                                train_end-train_start, sync_end-train_end))
                            self.f_acc.flush()
                        numpy.save('Logs/%d_%s_%s/params/param_client_%d_epoch_%d' % (self.savenum, self.dataset_model, self.is_iid, self.rank, epoch_id), list(self.model.parameters())[0][0][0].detach().cpu().numpy())
                        
            else:
                for step, (b_x, b_y) in enumerate(self.training_dataloader):
                    iter_id += 1
                    if CUDA:
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()
                    self.optimizer.zero_grad()
                    self.loss_func(self.model(b_x), b_y).backward()
                    output_loss = self.loss_func(self.model(b_x), b_y)
                    self.optimizer.step()
                    True_Flase, ratio = self.sync_manager.try_sync_model(iter_id)
                    time1 = time.time()-time0
                    if True_Flase: 
                        if self.rank == 0:
                            accuracy = self.test()
                            self.scheduler.step(accuracy)
                            self.logging(' - test - iter_id: %d; epoch_id: %d; round_id: %d; accuracy: %.4f; loss: %.4f;' % (iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy, output_loss))
                            self.f_acc.write('%.4f, %d, %d, %d, %.4f, %.4f, %.4f\n' % (time.time()-time0, iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy, output_loss, ratio))
                            self.f_acc.flush()
                        numpy.save('Logs/%d_%s_%s/params/param_client_%d_epoch_%d' % (self.savenum, self.dataset_model, self.is_iid, self.rank, epoch_id), list(self.model.parameters())[0][0][0].detach().cpu().numpy())
            self.logging('finish epoch: %d\n' % epoch_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--world_size', type=int, default=10)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--dataset_model', type=str, default='M1')
    parser.add_argument('--is_iid', type=str, default='F')
    
    args = parser.parse_args()
    if CUDA:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')
  
    sync_frequency = 5
    
    if args.dataset_model == 'M1':
        dataset_name = 'mnist'
        model_name = 'CNN_Mnist'
    if args.dataset_model == 'C1':
        dataset_name = 'cifar10'
        model_name = 'ResNet18_Cifar10'
    if args.dataset_model == 'W1':
        dataset_name = 'wikitext2'
        model_name = 'transformer'
    if args.dataset_model == 'E1':
        dataset_name = 'emnist'
        model_name = 'VGG11_EMNIST'
    
    if args.is_iid == 'T':
        is_iid = True
    if args.is_iid == 'F':
        is_iid = False
        
    local_training_profile = {
        'model_profile' : {
            'model_name' : model_name,
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'is_iid' : is_iid,
            'total_partition_number' : 1 if is_iid else args.world_size,
            'partition_rank' : 0 if is_iid else args.rank
        }
    }
    print('- Local Training Profile: ')
    pprint.pprint(local_training_profile)
    print

    sync_profile = {
        'sync_frequency' : sync_frequency,
        'interlayer_type' : 'GWPF',
        'dist_profile' : {
            'master_address' : args.master_address,
            'world_size' : args.world_size,
            'rank' : args.rank,
        },
    }
    print('- Sync Profile: ')
    pprint.pprint(sync_profile)
    print 

    client = Client(local_training_profile, sync_profile, args.trial_no, args.dataset_model, args.is_iid) 
    client.train()