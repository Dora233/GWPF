import torch
import copy, sys, numpy, math, datetime, pprint, time
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()
time_start = time.time()

InterlayerProfileDict = {
    'Default': None, 
    'GWPF': {
        'freezing_check_frequency' : 1, # unit for freezing_frequency is 'round'
        'ema_alpha' : 0.99,
        'stable_threshold_GWPF' : 0.05, #0.05
        'enable_random_freezing' : False
    },
    'APF': {
        'freezing_check_frequency' : 1, # unit for freezing_frequency is 'round'
        'ema_alpha' : 0.99,
        'stable_threshold' : 0.05, #0.05
        'enable_random_freezing' : False
    },
    'Gaia': {
        'initial_stable_threshold' : 0.01#0.01
    },
    'CMFL': {
        'initial_relevance_threshold' : 0.7
    },
}

class GWPF_Interlayer:
    def __init__(self, model, sync_frequency, apf_hyperparams, world_size):

        self.interlayer_type = 'GWPF'
        self.model = model
        self.sync_frequency = sync_frequency
        
        ''' Set hyper-parameters '''
        self.ema_alpha = apf_hyperparams['ema_alpha']
        self.stable_threshold = apf_hyperparams['stable_threshold_GWPF'] # this shall be larger than 1-self.ema_alpha
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = apf_hyperparams['enable_random_freezing']

        ''' Initialize statistics variables '''
        self.freezing_check_frequency = apf_hyperparams['freezing_check_frequency']  # the unit is one synchronization round
        self.world_size = world_size
        self.fc_round_id = 0  # freezing-check_round_id
        self.freezing_ratio = 0

        ''' Initialize model-structure-related variables '''
        #self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_param_length = -1 # this is indeed "self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])"
        self.init_model_structure_related_variables()

        ''' Initialize gwpf-algorithm-related statistics variables '''
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
        
        #self.signal = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()
        
        self.logging('create GWPF Interlayer', apf_hyperparams)

    def init_model_structure_related_variables(self):
        """ Initialize model-structure-related variables by flatten the model parameters one by one. """
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
        #return self.flattened_param_length

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [GWPF Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    """ Below are the public APIs callable by Sync_Manager. """
    def generate_tensor_list_to_transmit(self, iter_id):
        """ Create a list of tensor (in fact with only one element) to sync. """
        ''' flatten the parameters into a 1-dimension list. '''
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
            '''C = torch.cat( (A,B),0 ) dimension 0 vertical  C = torch.cat( (A,B),1 ) dimension 1 horizontal'''
        
        ''' Then roll back those should-be-frozen parameters. '''
        #self.tensor_of_flattened_params = torch.where(self.active_mask > 0, self.tensor_of_flattened_params, self.last_tensor_of_flattened_params)
        
        
        ''' If no synchronization, directly restore the model parameters and return. —— for normal params '''
        ''' directly restore the model parameters, accumulate updates(no roll back) and try to change mask   —— for frozen params'''
        if iter_id % self.sync_frequency != 0:
            for i, p in enumerate(self.model.parameters()):
                p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i]) # view---> resize
                #print('p.data1',numpy.array(torch.tensor(p.data,device='cpu')).shape)
            
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
            print('frozen_stepping_rate.shape',numpy.array(torch.tensor(frozen_stepping_rate,device='cpu')).shape) #decrease
            #print('frozen_stepping_rate',frozen_stepping_rate)
            print('frozen_stepping_rate value:',(sum(frozen_stepping_rate)/(numpy.array(torch.tensor(frozen_stepping_rate,device='cpu')).size)).item())
            #print('thaw threshold:',self.stable_threshold * self.world_size * self.sync_frequency) #0.5    * self.sync_frequency
            signal = torch.where(frozen_stepping_rate >= self.stable_threshold * self.world_size, torch.ones([frozen_size,1]).int().cuda(), torch.zeros([frozen_size,1]).int().cuda()) #* self.sync_frequency
            #print('signal worker value',signal)
            #print('worker need to thaw:',numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu')).shape)
            print('worker need to thaw:',len(numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu'))))
            time1 = time.time() - time_start
            dist.all_reduce(signal, op=dist.reduce_op.SUM, async_op=False)
            print('time for signal:', time.time() - time_start - time1)
            #print('signal server value',signal)
            #self.freezing_lengths2[self.freeze_index] = torch.where(signal > 0, self.freezing_lengths2[self.freeze_index]+1, torch.floor_divide(self.freezing_lengths2[self.freeze_index],2))
            self.freezing_lengths2[self.freeze_index] = torch.where(signal > 0, self.freezing_lengths2[self.freeze_index]+1, torch.floor_divide(self.freezing_lengths2[self.freeze_index],2))
            print('time for freezing_lengths2:', time.time() - time_start - time1)
            print('server need to thaw:',numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu')).shape)
        
        #if numpy.array(torch.tensor((self.freezing_lengths2 > 0).nonzero(),device='cpu')).size > self.active_mask.sum().float() * 0.5:
            print('Before refresh1:',self.active_mask.sum())
            self.active_mask = self.active_mask + (self.freezing_lengths2 > 0)
            print('After refresh1:',self.active_mask.sum())
            self.grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
            self.abs_grad_ema_frozen = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        
        
        ''' Then select those unfrozen parameters and package them into a new tensor for transmission'''
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
        #print('effective_stepping_rate.shape',numpy.array(torch.tensor(effective_stepping_rate,device='cpu')).shape) #decrease
        #print('effective_stepping_rate',effective_stepping_rate[10000:10050])
        #print('effective_stepping_rate value:',(sum(effective_stepping_rate)/(numpy.array(torch.tensor(effective_stepping_rate,device='cpu')).size)).item())
        #print('freeze threshold',self.stable_threshold) #0.05
        self.freezing_lengths[self.active_index] = torch.where(effective_stepping_rate < self.stable_threshold, self.freezing_lengths[self.active_index]+1, self.freezing_lengths[self.active_index]) #torch.floor_divide(x,2)
        print('Time3',time.time() - time_start)
        #print('freezing_lengths_1',self.freezing_lengths[10000:10050])
        #self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
        
        #print('Before refresh:',numpy.sum(numpy.array(torch.tensor(self.active_mask,device='cpu'))!=0))
        print('Before refresh:',self.active_mask.sum())
        #self.active_mask = (self.freezing_lengths == 0) + (self.freezing_lengths2 > 0)
        self.active_mask = (self.freezing_lengths == 0)
        #print('After refresh:',numpy.sum(numpy.array(torch.tensor(self.active_mask,device='cpu'))!=0)) # 11173962
        print('After refresh:',self.active_mask.sum()) # tensor(11173962, device='cuda:0')
        self.active_index = self.active_mask.nonzero()
        self.freeze_mask = (self.active_mask == False)
        self.freeze_index = self.freeze_mask.nonzero()
        print('freeze_index.shape',numpy.array(torch.tensor(self.freeze_index,device='cpu')).shape)
        print('num of frozen',self.active_mask.sum().float())
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_param_length
        self.logging('current stable ratio: %.4f' % self.freezing_ratio)
        if self.freezing_ratio > self.tighter_stable_criterion_threshold:
            self.stable_threshold /= 2.0
            self.logging('make stable criterion tighter')
        return self.freezing_ratio


class APF_Interlayer:
    def __init__(self, model, sync_frequency, apf_hyperparams):

        self.interlayer_type = 'APF'
        self.model = model
        self.sync_frequency = sync_frequency

        ''' Set hyper-parameters '''
        self.ema_alpha = apf_hyperparams['ema_alpha']
        self.stable_threshold = apf_hyperparams['stable_threshold'] # this shall be larger than 1-self.ema_alpha
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = apf_hyperparams['enable_random_freezing']

        ''' Initialize statistics variables '''
        self.freezing_check_frequency = apf_hyperparams['freezing_check_frequency']  # the unit is one synchronization round
        self.fc_round_id = 0  # freezing-check_round_id
        self.freezing_ratio = 0

        ''' Initialize model-structure-related variables '''
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_param_length = -1 # this is indeed "self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])"
        self.init_model_structure_related_variables()

        ''' Initialize apf-algorithm-related statistics variables '''
        self.active_mask = torch.ones(self.flattened_param_length).byte().cuda() if CUDA else torch.ones(self.flattened_param_length).byte()
        self.active_index = self.active_mask.nonzero()
        self.grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.abs_grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.freezing_lengths = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()
        self.fc_round_ids_to_unfreeze_params = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()

        self.logging('create APF Interlayer', apf_hyperparams)

    def init_model_structure_related_variables(self):
        """ Initialize model-structure-related variables by flatten the model parameters one by one. """
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

    def logging(self, string, hyperparameters=None): # each class shall have a 'logging' function right after initialization function
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [APF Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    """ Below are the public APIs callable by Sync_Manager. """
    def generate_tensor_list_to_transmit(self, iter_id):
        """ Create a list of tensor (in fact with only one element) to sync. """
        ''' flatten the parameters into a 1-dimension list. '''
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
            ''' Then roll back those should-be-frozen parameters. '''
        #print('tensor_of_flattened_params',numpy.array(torch.tensor(self.tensor_of_flattened_params,device='cpu')).shape) #tensor_of_flattened_params (11173962,)
        #print('active_mask',numpy.array(torch.tensor(self.active_mask,device='cpu')).shape) #active_mask (11173962,)
        self.tensor_of_flattened_params = torch.where(self.active_mask > 0, self.tensor_of_flattened_params, self.last_tensor_of_flattened_params)
        ''' If no synchronization, directly restore the model parameters and return. '''
        if iter_id % self.sync_frequency != 0:
            for i, p in enumerate(self.model.parameters()):
                p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
                #print('p.data1',numpy.array(torch.tensor(p.data,device='cpu')).shape)
            return []
        ''' Then select those unfrozen parameters and package them into a new tensor for transmission'''
        tensor_to_transmit = torch.masked_select(self.tensor_of_flattened_params, self.active_mask)
        return [tensor_to_transmit]

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.tensor_of_flattened_params[self.active_mask] = tensor_list_received[0]  # for APF, there should be only one element in tensor_list_received
        ''' Unflattern parameters to model parameters. '''
        for i, p in enumerate(self.model.parameters()):
            p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
            #print('p.data2',numpy.array(torch.tensor(p.data,device='cpu')).shape)
        ''' Refresh the active mask after one synchronization round finishes '''
        freezing_ratio = self.refresh_freezing_status()
        return freezing_ratio

    def refresh_freezing_status(self):
        """ Called after each global synchronization is conducted. Update the freezing status and related statistics."""
        ''' Update the freezing period in a TCP manner and then update the freezing mask accordingly '''
        self.update_param_freezing_periods_in_TCP_manner()
        self.fc_round_id += 1
        print('self.fc_round_id',self.fc_round_id)
        
        active_mask2 = numpy.array(torch.tensor(self.active_mask,device='cpu'))
        print('Before refresh:',numpy.sum(active_mask2!=0))
        self.active_mask = (self.fc_round_ids_to_unfreeze_params == self.fc_round_id) # (11173962,) all the time
        active_mask3 = numpy.array(torch.tensor(self.active_mask,device='cpu'))
        print('After refresh:',numpy.sum(active_mask3!=0))
        print('self.active_mask[a:b]',self.active_mask[10000:10050])
        #print('active_mask2',numpy.array(torch.tensor(self.active_mask,device='cpu')).shape) #active_mask2 (11173962,)
        
        ''' Record the stable-ratio statistics and adaptively tune stability threshold if necessary '''
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_param_length
        self.logging('current stable ratio: %.4f' % self.freezing_ratio)
        if self.freezing_ratio > self.tighter_stable_criterion_threshold:
            self.stable_threshold /= 2.0
            self.logging('make stable criterion tighter')
        return self.freezing_ratio

    def update_param_freezing_periods_in_TCP_manner(self):
        """ update the active mask of each parameter based on global gradient stability. """
        ''' Calculate the effective stepping rate of each parameter (in tensor mode for fast processing). '''
        self.active_index = self.active_mask.nonzero()
        self.grad = self.last_tensor_of_flattened_params - self.tensor_of_flattened_params
        self.grad_ema[self.active_index] = self.grad_ema[self.active_index] * self.ema_alpha + self.grad[self.active_index] * (1.0 - self.ema_alpha) 
        self.abs_grad_ema[self.active_index] = self.abs_grad_ema[self.active_index] * self.ema_alpha + torch.abs(self.grad[self.active_index]) * (1.0 - self.ema_alpha)
        effective_stepping_rate = torch.abs(self.grad_ema[self.active_index]) / self.abs_grad_ema[self.active_index]
        #print('effective_stepping_rate',effective_stepping_rate[10000:10050])
        ''' Update the freezing period length and also the should-be-unfrozen frequency_check_round_id based on effective stepping rate of each param. '''
        print('Time1:',time.time() - time_start)
        self.freezing_lengths[self.active_index] = torch.where(effective_stepping_rate < self.stable_threshold, self.freezing_lengths[self.active_index]+1, torch.floor_divide(self.freezing_lengths[self.active_index],2))
        print('Time2:',time.time() - time_start)
        #print('self.freezing_lengths[a:b]',self.freezing_lengths[10000:10050])
        self.fc_round_ids_to_unfreeze_params[self.active_index] = self.fc_round_id + self.freezing_lengths[self.active_index] + 1
        print('self.fc_round_ids_to_unfreeze_params[a:b]',self.fc_round_ids_to_unfreeze_params[10000:10050])
        if self.enable_random_freezing:
            self.randomly_freeze_active_params(self.active_index)
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)

    def randomly_freeze_active_params(self):
        rand_array = torch.rand(self.active_index.shape) * 100
        rand_frozen = torch.where(rand_array < self.fc_round_id / 20.0, rand_array.int(), torch.zeros(rand_array.shape).int())
        rand_frozen = rand_frozen.cuda() if CUDA else rand_frozen 
        self.fc_round_ids_to_unfreeze_params[self.active_index] = self.fc_round_ids_to_unfreeze_params[self.active_index] + rand_frozen 


class CMFL_Interlayer:
    def __init__(self, model, sync_frequency, cmfl_hyperparams):
        self.interlayer_type = 'CMFL'
        self.model = model
        self.sync_frequency = sync_frequency
        self.total_param_num = sum([float(p.data.nelement()) for p in self.model.parameters()])
        self.initial_relevance_threshold = cmfl_hyperparams['initial_relevance_threshold']
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]
        self.last_global_update_list = [torch.zeros(p.data.shape).cuda() if CUDA else torch.zeros(p.data.shape) for p in self.model.parameters()]
        self.logging('create CMFL Interlayer', cmfl_hyperparams)
    
    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [CMFL Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print 
        sys.stdout.flush()

    def generate_tensor_list_to_transmit(self, iter_id):
        """ Faked version. Instead of excluding self from all-reduce, packing initial param into tensor_list """
        if iter_id % self.sync_frequency != 0:
            return []
        relevance_threshold = self.initial_relevance_threshold / math.sqrt(iter_id/100000.0+1.0)
        #relevance_threshold = self.initial_relevance_threshold
        current_relevance = self.get_relevance_to_global_update()
        self.logging('relevance threshold: %.4f; current relevance: %.4f; shall_report: %s'  % (relevance_threshold, current_relevance, 'True' if current_relevance > relevance_threshold else 'False'))
        if current_relevance > relevance_threshold:
            ''' If relevant to global update, then report local update to parameter server. '''
            tensor_list_to_transmit = [p.data for p in self.model.parameters()]
        else:
            ''' Else report initial parameters to PS. '''
            tensor_list_to_transmit = copy.deepcopy(self.last_param_list)
        return tensor_list_to_transmit

    def get_relevance_to_global_update(self):
        relevant_element_number = 0
        for i, param in enumerate(self.model.parameters()):
            relevance_tensor = torch.where((param.data-self.last_param_list[i]) * self.last_global_update_list[i] >= 0, \
                torch.ones(param.data.shape).cuda() if CUDA else torch.ones(param.data.shape), torch.zeros(param.data.shape).cuda() if CUDA else torch.zeros(param.data.shape))
            relevant_element_number += float(relevance_tensor.sum())
        relevance = float(relevant_element_number) / self.total_param_num
        return relevance

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.last_global_update_list = []
        index = 0
        for tensor_i in tensor_list_received:
            print('tensor_i',numpy.array(torch.tensor(tensor_i,device='cpu')).shape)
            index += 1
            '''tensor_i (64, 3, 3, 3)
                tensor_i (64,)
                tensor_i (192, 64, 3, 3)
                tensor_i (192,)
                tensor_i (384, 192, 3, 3)
                tensor_i (384,)
                tensor_i (256, 384, 3, 3)
                tensor_i (256,)
                tensor_i (256, 256, 3, 3)
                tensor_i (256,)
                tensor_i (4096, 1024)
                tensor_i (4096,)
                tensor_i (4096, 4096)
                tensor_i (4096,)
                tensor_i (10, 4096)
                tensor_i (10,)'''
        for i, param in enumerate(self.model.parameters()):
            param.data = tensor_list_received[i]
            self.last_global_update_list.append(tensor_list_received[i] - self.last_param_list[i])
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]


class Gaia_v2_Interlayer:
    def __init__(self, model, sync_frequency, gaia_hyperparams):
        self.interlayer_type = 'Gaia'
        self.model = model
        self.sync_frequency = sync_frequency
        self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])
        self.initial_stable_threshold = gaia_hyperparams['initial_stable_threshold']
        self.significant_mask_list = [torch.ones(p.data.shape).byte().cuda() if CUDA else torch.ones(p.data.shape).byte() for p in self.model.parameters()]
        self.significant_ratio = 1.0
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]

        ''' Initialize model-structure-related variables '''
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_param_length = -1 # this is indeed "self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])"
        self.init_model_structure_related_variables()
        self.active_mask = torch.ones(self.flattened_param_length).byte().cuda() if CUDA else torch.ones(self.flattened_param_length).byte()
        
        self.iter_id = 0
        self.logging('create Gaia Interlayer', gaia_hyperparams)

    def init_model_structure_related_variables(self):
        """ Initialize model-structure-related variables by flatten the model parameters one by one. """
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
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [Gaia Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print 
        sys.stdout.flush()

    def generate_tensor_list_to_transmit(self, iter_id):
        self.iter_id = iter_id
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        ''' Then roll back those should-be-frozen parameters. '''
        self.tensor_of_flattened_params = torch.where(self.active_mask > 0, self.tensor_of_flattened_params, self.last_tensor_of_flattened_params)
        
        """ Roll back those locally-stable parameters to their previous values. """
        """ TODO: Current the communication reduction is faked. """
        if iter_id % self.sync_frequency != 0:
            for i, p in enumerate(self.model.parameters()):
                p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
            return []
        
        tensor_to_transmit = torch.masked_select(self.tensor_of_flattened_params, self.active_mask)
        return [tensor_to_transmit]

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.tensor_of_flattened_params[self.active_mask] = tensor_list_received[0]
        for i, p in enumerate(self.model.parameters()):
            p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
            #param.data = torch.where(self.significant_mask_list[i], tensor_list_received[i], param.data)
        #self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]
        
        #tensor_list_to_transmit = []
        # stable_threshold = self.initial_stable_threshold / math.sqrt(iter_id)
        stable_threshold = self.initial_stable_threshold / math.sqrt(self.iter_id/10000.0+1.0)
        # for i, param in enumerate(self.model.parameters()):
            # self.significant_mask_list[i] = (torch.abs(self.last_param_list[i]/param.data - 1) > stable_threshold)
            # tensor_to_transmit = torch.where(self.significant_mask_list[i], param.data, self.last_param_list[i])
            # tensor_list_to_transmit.append(tensor_to_transmit)
        # self.significant_ratio = sum([float(significant_mask.sum()) for significant_mask in self.significant_mask_list]) / self.total_param_num
        
        self.grad = self.last_tensor_of_flattened_params - self.tensor_of_flattened_params
        self.divide = torch.abs(self.grad/self.tensor_of_flattened_params)
        print('divide',self.divide[10000:10050])
        '''divide tensor([8.1252e-03, 3.7196e-03, 5.4360e-03, 1.2169e-02, 2.5396e-03, 4.6564e-03,
        1.1969e-02, 1.2550e-03, 2.1839e-03, 2.6371e-03, 1.3763e-03, 2.0442e-04,
        6.1880e-04, 1.5757e-03, 1.6339e-03, 2.9519e-03, 3.7140e-03, 4.4159e-03,
        2.4719e-03, 3.1955e-03, 3.7816e-02, 3.5347e-03, 2.0272e-03, 3.6684e-03,
        5.3523e-03, 3.2046e-03, 2.6164e-03, 3.5761e-03, 1.9319e-02, 7.0059e-03,
        3.5207e-05, 2.5192e-03, 1.4422e-02, 3.0966e-02, 1.6852e-03, 2.1318e-03,
        2.8648e-04, 6.3027e-04, 7.3868e-05, 6.5426e-04, 8.9031e-04, 2.9318e-03,
        2.2227e-03, 1.6681e-03, 2.8859e-03, 4.0203e-03, 2.5562e-03, 7.7059e-04,
        3.1863e-03, 8.1300e-04], device='cuda:0')'''
        self.active_mask = (self.divide > stable_threshold)
        #print('active_mask',numpy.sum(self.active_mask!=0))
        print('active_mask',numpy.array(torch.tensor(self.active_mask,device='cpu')).shape) # (11173962,)
        print('active_mask[a:b]',self.active_mask[10000:10050])
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
        self.significant_ratio = self.active_mask.sum().float() / self.flattened_param_length
        #print('total_param_num',self.total_param_num) #11173962
        #print('flattened_param_length',self.flattened_param_length) #11173962
        self.logging('current significant ratio: %.4f' % self.significant_ratio)


class Gaia_Interlayer:
    def __init__(self, model, sync_frequency, gaia_hyperparams):
        self.interlayer_type = 'Gaia'
        self.model = model
        self.sync_frequency = sync_frequency
        self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])
        self.initial_stable_threshold = gaia_hyperparams['initial_stable_threshold']
        self.significant_mask_list = [torch.ones(p.data.shape).byte().cuda() if CUDA else torch.ones(p.data.shape).byte() for p in self.model.parameters()]
        self.significant_ratio = 1.0
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]

        self.logging('create Gaia Interlayer', gaia_hyperparams)

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [Gaia Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print 
        sys.stdout.flush()

    def generate_tensor_list_to_transmit(self, iter_id):
        """ Roll back those locally-stable parameters to their previous values. """
        """ TODO: Current the communication reduction is faked. """
        if iter_id % self.sync_frequency != 0:
            return []
        tensor_list_to_transmit = []
        # stable_threshold = self.initial_stable_threshold / math.sqrt(iter_id)
        stable_threshold = self.initial_stable_threshold / math.sqrt(iter_id/10000.0+1.0)
        for i, param in enumerate(self.model.parameters()):
            self.significant_mask_list[i] = (torch.abs(self.last_param_list[i]/param.data - 1) > stable_threshold)
            tensor_to_transmit = torch.where(self.significant_mask_list[i], param.data, self.last_param_list[i])
            tensor_list_to_transmit.append(tensor_to_transmit)
        self.significant_ratio = sum([float(significant_mask.sum()) for significant_mask in self.significant_mask_list]) / self.total_param_num
        self.logging('current significant ratio: %.4f' % self.significant_ratio)
        return tensor_list_to_transmit

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.where(self.significant_mask_list[i], tensor_list_received[i], param.data)
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]


class Default_Interlayer:
    def __init__(self, model, sync_frequency):
        self.interlayer_type = 'Default'
        self.model = model
        self.sync_frequency = sync_frequency
        self.logging('create Default Interlayer')
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [Default Interlayer] '+str(string))
        sys.stdout.flush()
    
    def generate_tensor_list_to_transmit(self, iter_id):
        if iter_id % self.sync_frequency != 0:
            return []
        tensor_list_to_transmit = []
        #index = 0
        for param in self.model.parameters():
            tensor_list_to_transmit.append(param.data)
            #index += 1
            #print('index',index) # 1:62
        return tensor_list_to_transmit
    
    def restore_model_from_tensor_list_received(self, tensor_list_received):
        # index = 0
        # for tensor in tensor_list_received:
            # tensor_list[i] = tensor
            # index += 1
        
        for i, param in enumerate(self.model.parameters()):
            # Note that you are actually assigning a variable to itself
            #print('i',i)
            param.data = tensor_list_received[i]


class Sync_Manager:
    """ This object shall be able to run without apf (support standard FL by default) """
    def __init__(self, model, sync_profile):
        self.model = model
        self.sync_round_id = 0
        self.world_size = 0  # world_size is required in all_reduce averaging.
        self.sync_frequency = sync_profile['sync_frequency']
        print('self.world_size1',self.world_size)
        self.init_dist(sync_profile['dist_profile'])
        print('self.world_size2',self.world_size)
        self.transmit_interlayer = self.create_transmit_interlayer(sync_profile['interlayer_type'])
        self.interlayer_type = sync_profile['interlayer_type']
        

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()

    def create_transmit_interlayer(self, interlayer_type):
        if interlayer_type == 'Default':
            transmit_interlayer = Default_Interlayer(self.model, self.sync_frequency)
        if interlayer_type == 'APF':
            transmit_interlayer = APF_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        if interlayer_type == 'Gaia':
            transmit_interlayer = Gaia_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        if interlayer_type == 'CMFL':
            transmit_interlayer = CMFL_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        if interlayer_type == 'GWPF':
            transmit_interlayer = GWPF_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type], self.world_size)
        return transmit_interlayer

    def init_dist(self, dist_profile):
        self.world_size = dist_profile['world_size']  # world_size is required in all_reduce averaging.
        dist.init_process_group(backend='nccl' if CUDA else 'tcp', init_method=dist_profile['master_address'], world_size=dist_profile['world_size'], rank=dist_profile['rank'])
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def try_sync_model(self, iter_id):
        freezing_ratio = 0 
        time1 = time.time() - time_start
        tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        print('time for generate:', time.time() - time_start - time1)
        ''' Conduct remote synchronization. '''
        if tensor_list_to_transmit != []:
            self.sync_round_id += 1
            tensor_list_received = []
            index = 0
            time2 = time.time() - time_start
            for tensor_to_transmit in tensor_list_to_transmit:
                dist.all_reduce(tensor_to_transmit, op=dist.reduce_op.SUM, async_op=False)  # transmit parameter    async_op = True----async
                tensor_list_received.append(tensor_to_transmit / self.world_size)  # receive parameter
            #print('tensor_to_transmit',tensor_to_transmit) #tensor([-0.1398, -0.0050, -0.0237,  ...,  0.0347,  0.0145,  0.0147], device='cuda:0')
            print('tensor_to_transmit',numpy.array(torch.tensor(tensor_to_transmit,device='cpu')).shape)
            #print('tensor_to_transmit size',sys.getsizeof(tensor_to_transmit))  #64B
            print('time for tensor:', time.time() - time_start - time2)
                
            if self.interlayer_type == 'APF' or 'GWPF': 
                freezing_ratio = self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
                print('time for restore:', time.time() - time_start - time2)
                return True,freezing_ratio
            else:
                self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
                return True
    
        if self.interlayer_type == 'APF' or 'GWPF':
            return False,freezing_ratio
        else:
            return False