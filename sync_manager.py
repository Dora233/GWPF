import torch
import copy, sys, numpy, math, datetime, pprint, time
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()
time_start = time.time()

InterlayerProfileDict = {
    'GWPF': {
        'freezing_check_frequency' : 1,
        'ema_alpha' : 0.99,
        'stable_threshold_GWPF' : 0.02,
        'enable_random_freezing' : False
    }
}

class GWPF_Interlayer:
    def __init__(self, model, sync_frequency, apf_hyperparams, world_size):

        self.interlayer_type = 'GWPF'
        self.model = model
        self.sync_frequency = sync_frequency

        self.ema_alpha = apf_hyperparams['ema_alpha']
        self.stable_threshold = apf_hyperparams['stable_threshold_GWPF']
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = apf_hyperparams['enable_random_freezing']

        self.freezing_check_frequency = apf_hyperparams['freezing_check_frequency']
        self.world_size = world_size
        self.fc_round_id = 0 
        self.freezing_ratio = 0

        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_param_length = -1 
        self.init_model_structure_related_variables()

        self.active_mask = torch.ones(self.flattened_param_length).byte().cuda() if CUDA else torch.ones(self.flattened_param_length).byte()
        self.freeze_mask = torch.zeros(self.flattened_param_length).byte().cuda() if CUDA else torch.zeros(self.flattened_param_length).byte()
        self.active_index = self.active_mask.nonzero()
        self.freeze_index = self.freeze_mask.nonzero()
        
        self.grad = self.last_tensor_of_flattened_params
        self.grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.abs_grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        
        self.freezing_lengths = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()
        self.freezing_lengths2 = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()

        self.grad_frozen = self.last_tensor_of_flattened_params
        self.grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.abs_grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        
        self.logging('create GWPF Interlayer', apf_hyperparams)

    def init_model_structure_related_variables(self):
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            self.param_shape_list.append(p.data.shape)
            flattened_param = p.data.view(-1)
            e_index = s_index + len(flattened_param)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        self.flattened_param_length = len(self.tensor_of_flattened_params)
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [GWPF Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    def generate_tensor_list_to_transmit(self, iter_id):
        """ Create a list of tensor (in fact with only one element) to sync. """
        ''' flatten the parameters into a 1-dimension list. '''
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)

        ''' If no synchronization, directly restore the model parameters and return. —— for normal params '''
        ''' directly restore the model parameters, accumulate updates(no roll back) and try to change mask   —— for frozen params'''
        if iter_id % self.sync_frequency != 0:
            for i, p in enumerate(self.model.parameters()):
                p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i]) 
            
            if numpy.array(torch.tensor(self.freeze_index,device='cpu')).size > 0:
                self.grad_frozen[self.freeze_index] = self.last_tensor_of_flattened_params[self.freeze_index] - self.tensor_of_flattened_params[self.freeze_index]
                self.grad_ema_frozen[self.freeze_index] = self.grad_ema_frozen[self.freeze_index] * self.ema_alpha + self.grad_frozen[self.freeze_index] * (1.0 - self.ema_alpha) 
                self.abs_grad_ema_frozen[self.freeze_index] =  self.abs_grad_ema_frozen[self.freeze_index] * self.ema_alpha + torch.abs(self.grad_frozen[self.freeze_index]) * (1.0 - self.ema_alpha) 
                
                self.last_tensor_of_flattened_params[self.freeze_index] = copy.deepcopy(self.tensor_of_flattened_params[self.freeze_index])
            return []
        
        frozen_size = numpy.array(torch.tensor(self.freeze_index,device='cpu')).size
        print('frozen size',frozen_size)
        if frozen_size > 0:
            frozen_stepping_rate = torch.abs(self.grad_ema_frozen[self.freeze_index]) / self.abs_grad_ema_frozen[self.freeze_index]
            print('frozen_stepping_rate.shape',numpy.array(torch.tensor(frozen_stepping_rate,device='cpu')).shape)
            print('frozen_stepping_rate value:',(sum(frozen_stepping_rate)/(numpy.array(torch.tensor(frozen_stepping_rate,device='cpu')).size)).item())
            signal = torch.where(frozen_stepping_rate >= self.stable_threshold, torch.ones([frozen_size,1]).int().cuda(), torch.zeros([frozen_size,1]).int().cuda())
            print('worker need to thaw:',len(numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu'))))
            time1 = time.time() - time_start
            dist.all_reduce(signal, op=dist.reduce_op.SUM, async_op=False)
            print('time for signal:', time.time() - time_start - time1)
            self.freezing_lengths2[self.freeze_index] = torch.where(signal > 0, self.freezing_lengths2[self.freeze_index]+1, torch.floor_divide(self.freezing_lengths2[self.freeze_index],2))
            print('time for freezing_lengths2:', time.time() - time_start - time1)
            print('server need to thaw:',numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu')).shape)
        
            print('Before refresh1:',self.active_mask.sum())
            self.active_mask = self.active_mask + (self.freezing_lengths2 > 0)
            print('After refresh1:',self.active_mask.sum())
            self.grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
            self.abs_grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        
        ''' Then select those normal parameters and package them into a new tensor for transmission'''
        tensor_to_transmit = torch.masked_select(self.tensor_of_flattened_params, self.active_mask)
        return [tensor_to_transmit]
        

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.tensor_of_flattened_params[self.active_mask] = tensor_list_received[0]
        self.fc_round_id += 1
        print('Time:%f, sync_round:%d' % (time.time() - time_start,self.fc_round_id))
        ''' Unflattern parameters to model parameters. '''
        for i, p in enumerate(self.model.parameters()):
            p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
        
        print('Time1',time.time() - time_start)
        self.grad[self.active_index] = self.last_tensor_of_flattened_params[self.active_index] - self.tensor_of_flattened_params[self.active_index]
        self.last_tensor_of_flattened_params[self.active_index] = copy.deepcopy(self.tensor_of_flattened_params[self.active_index])
        
        self.grad_ema[self.active_index] = self.grad_ema[self.active_index] * self.ema_alpha + self.grad[self.active_index] * (1.0 - self.ema_alpha) 
        self.abs_grad_ema[self.active_index] = self.abs_grad_ema[self.active_index] * self.ema_alpha + torch.abs(self.grad[self.active_index]) * (1.0 - self.ema_alpha)
        effective_stepping_rate = torch.abs(self.grad_ema[self.active_index]) / self.abs_grad_ema[self.active_index]
        print('Time2',time.time() - time_start)
        self.freezing_lengths[self.active_index] = torch.where(effective_stepping_rate < self.stable_threshold, self.freezing_lengths[self.active_index]+1, self.freezing_lengths[self.active_index]) #torch.floor_divide(x,2)
        print('Time3',time.time() - time_start)
        
        print('Before refresh:',self.active_mask.sum())
        self.active_mask = (self.freezing_lengths == 0)
        print('After refresh:',self.active_mask.sum())
        self.active_index = self.active_mask.nonzero()
        self.freeze_mask = (self.active_mask == False)
        self.freeze_index = self.freeze_mask.nonzero()
        print('freeze_index.shape',numpy.array(torch.tensor(self.freeze_index,device='cpu')).shape)
        print('num of frozen',self.active_mask.sum().float())
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_param_length
        self.logging('current stable ratio: %.4f' % self.freezing_ratio)
        if self.freezing_ratio > self.tighter_stable_criterion_threshold:
            self.stable_threshold /= 2.0
        return self.freezing_ratio

class Sync_Manager:
    def __init__(self, model, sync_profile):
        self.model = model
        self.sync_round_id = 0
        self.world_size = 0 
        self.sync_frequency = sync_profile['sync_frequency']
        print('self.world_size1',self.world_size)
        self.init_dist(sync_profile['dist_profile'])
        print('self.world_size2',self.world_size)
        self.transmit_interlayer = GWPF_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type], self.world_size)
        self.interlayer_type = sync_profile['interlayer_type']

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()

    def init_dist(self, dist_profile):
        self.world_size = dist_profile['world_size']
        dist.init_process_group(backend='nccl' if CUDA else 'tcp', init_method=dist_profile['master_address'], world_size=dist_profile['world_size'], rank=dist_profile['rank'])
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def try_sync_model(self, iter_id):
        freezing_ratio = 0 
        time1 = time.time() - time_start
        tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        print('time for generate:', time.time() - time_start - time1)
        if tensor_list_to_transmit != []:
            self.sync_round_id += 1
            tensor_list_received = []
            index = 0
            time2 = time.time() - time_start
            for tensor_to_transmit in tensor_list_to_transmit:
                dist.all_reduce(tensor_to_transmit, op=dist.reduce_op.SUM, async_op=False)  # transmit parameter
                tensor_list_received.append(tensor_to_transmit / self.world_size)  # receive parameter
            print('tensor_to_transmit',numpy.array(torch.tensor(tensor_to_transmit,device='cpu')).shape)
            print('time for tensor:', time.time() - time_start - time2)
                
            freezing_ratio = self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
            print('time for restore:', time.time() - time_start - time2)
            return True,freezing_ratio

        return False,freezing_ratio