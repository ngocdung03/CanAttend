import copy
import json
import math
import os
import pdb
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from .losses import NLLPCHazardLoss
from .utils import de_tuple


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError("Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    
############################
# optimizer #
############################

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        print("l_total=",len(self.param_groups))
        for group in self.param_groups:
            print("l_p=",len(group['params']))
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                pdb.set_trace()
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


############################
# trainer #
############################

class Trainer:
    def __init__(self, model, metrics=None, verbose=True, weighting_tag=True, train_by_val_stats=False, device=None):
        '''metrics must start from NLLPCHazardLoss, then be others
        Arguments:
            model: an instance of CA model.
            metrics (func): list of loss functions.
            verbose (bool): whether to print verbose on train and val losses.
            weighting_tag (bool): whether to adjust transference on weights (eg. 1/cases).
        '''
        self.model = model # Should be CAMulti for TAG
        if metrics is None:
            self.metrics = [NLLPCHazardLoss(),]

        self.train_logs = defaultdict(list)
        # self.get_target = lambda df: (df['duration'].values, df['event'].values)
        self.use_gpu = True if torch.cuda.is_available() else False
        if self.use_gpu:
            print('use pytorch-cuda for training.')
            self.model.cuda()
            self.model.use_gpu = True
            self.device = 'cuda'
        else:
            print('GPU not found! will use cpu for training!')
            self.device = 'cpu'
            
        if device:
            self.device = device
        self.early_stopping = None
        ckpt_dir = os.path.dirname(model.config['checkpoint'])
        self.ckpt = model.config['checkpoint']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.encoder_trainble = [
                p for p in self.model.encoder.parameters() if p.requires_grad
        ] ## TAG Alter
        self.verbose = verbose ##
        self.weighting_tag = weighting_tag
        self.train_by_val_stats = train_by_val_stats
            
    ## TAG
    def lookahead(self, x, loss, optimizer, event):
        batch_x_cat, batch_x_num = x ##
        optimizer.zero_grad()
        shared_params = self.encoder_trainble
        init_weights = [param.data for param in shared_params]
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
        
        # Compute updated params for the forward pass: SGD w/ 0.9 momentum + 1e-4 weight decay.
        opt_state = optimizer.state_dict()['param_groups'][0] ##
        weight_decay = opt_state['weight_decay']
        
        for param, grad, param_id in zip(shared_params, grads, opt_state['params']):
        
            grad += param.data * weight_decay ## param
            if 'momentum_buffer' not in opt_state:
                mom_buf = grad
            else:
                mom_buf = opt_state['momentum_buffer']
                mom_buf = mom_buf * opt_state['momentum'] + grad
            param.data = param.data - opt_state['lr'] * mom_buf

        with torch.no_grad():
            output = self.model(input_ids=batch_x_cat, input_nums=batch_x_num, event=event) ##

        for param, init_weight in zip(shared_params, init_weights):
            param.data = init_weight
        return output
    ##

    def train_single_event(self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):
        
        df_train, df_y_train = train_set
        # durations_train, events_train = self.get_target(df_y_train)

        if val_set is not None:
            df_val, df_y_val = val_set
            # durations_val, events_val = self.get_target(df_y_val)
            tensor_val = torch.tensor(val_set[0].values)
            tensor_y_val = torch.tensor(val_set[1].values)
        
        if self.use_gpu:
            tensor_val = tensor_val.cuda()
            tensor_y_val = tensor_y_val.cuda()

        # assign no weight decay on these parameters
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters, 
            learning_rate, 
            weight_decay_rate=weight_decay, 
            )

        if val_set is not None:
            # take early stopping
            self.early_stopping = EarlyStopping(patience=self.model.config['early_stop_patience'])

        num_train_batch = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            df_train = train_set[0].sample(frac=1)
            df_y_train = train_set[1].loc[df_train.index]

            tensor_train = torch.tensor(df_train.values)
            tensor_y_train = torch.tensor(df_y_train.values)
            if self.use_gpu:
                tensor_y_train = tensor_y_train.cuda()
                tensor_train = tensor_train.cuda()

            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                
                if len(batch_train) < batch_size: ## Introduce drop_last=True
                    pass
                
                else: 
                    batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                    batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()
                    
                    phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num)

                    if len(self.metrics) == 1: # only NLLPCHazardLoss is asigned
                        batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float(), reduction="mean")

                    else:
                        raise NotImplementedError                    
                    batch_loss.backward()
                    optimizer.step()

                    epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx+1))

            if val_set is not None:
                self.model.eval()
                with torch.no_grad():
                    phi_val = de_tuple(self.model.predict(tensor_val, val_batch_size)) ##
                
                val_loss = self.metrics[0](phi_val, tensor_y_val[:,0].long(), tensor_y_val[:,1].long(), tensor_y_val[:,2].float())
                print("[Train-{}]: {}".format(epoch, epoch_loss))
                print("[Val-{}]: {}".format(epoch, val_loss.item()))
                val_loss_list.append(val_loss.item())
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    print(f"early stops at epoch {epoch+1}")
                    # load best checkpoint
                    self.model.load_state_dict(torch.load(self.ckpt))
                    return train_loss_list, val_loss_list
            else:
                print("[Train-{}]: {}".format(epoch, epoch_loss))

        return train_loss_list, val_loss_list

    def train_multi_event(self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        criterion_fun=None, ## TAG
        **kwargs,
        ):

        task_name = [] #"duration", "proportion"
        for risk in range(self.model.config.num_event):
            task_name.append("event_{}".format(risk))
        if val_set is not None:
            tensor_val = torch.tensor(val_set[0].values).to(self.device)
            # MOD
            tensor_y_val = torch.tensor(val_set[1][['duration', 'proportion'] + task_name].values).to(self.device)
            ## 
            
        # assign no weight decay on these parameters
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters, 
            learning_rate, 
            weight_decay_rate=weight_decay, 
            )
        if val_set is not None:
            # take early stopping
            self.early_stopping = EarlyStopping(patience=self.model.config['early_stop_patience'])

        train_loss_list, val_loss_list = [], []
        num_train_batch = int(np.ceil(len(train_set[0]) / batch_size))
        
        ## TAG
        
        transference = {combined_task: [] for combined_task in task_name} #self.loss_keys
        # best_val = 1000.
        train_dict = {epoch: {} for epoch in range(epochs)}
        valid_dict = {epoch: {} for epoch in range(epochs)}
        # test_dict = {epoch: {} for epoch in range(self.start_epoch, FLAGS.epochs)}
        df_train, df_y_train = train_set[0], train_set[1]
        case_weights = (1/df_y_train[task_name].sum()).to_list() #torch.tensor(1/df_y_train[task_name].sum()).to(self.device) # !! track loss gradient?
        ## 
        for epoch in range(epochs):
            df_train = df_train.sample(frac=1)
            df_y_train = df_y_train.loc[df_train.index] 

            tensor_train = torch.tensor(df_train.values).to(self.device)
            
            tensor_y_train = torch.tensor(df_y_train[['duration', 'proportion'] + task_name].values).to(self.device) ## MOD, Ensure the order
            
            epoch_loss = 0
            self.model.train() ## Why did not for multi_event
            
            ## TAG train_epoch()
            average_meters = defaultdict(AverageMeter)
            epoch_transference = {}
            for combined_task in task_name: #self.loss_keys
                epoch_transference[combined_task] = {}
                for recipient_task in task_name: #self.loss_keys
                    epoch_transference[combined_task][recipient_task] = 0.
            ##
            ## Train and Loss - TAG train_epoch injected
            for batch_idx in range(num_train_batch):
                
                optimizer.zero_grad() # 0924

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]

                batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()
                
                
                ## TAG
                batch_loss = 0
                loss_dict = None
                
                loss_dict = {} ## Standardize loss by case weights
                
                for risk in range(self.model.config.num_event): #
                    phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num, event=risk)
                    batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    # phi[1].size()) [N, 4] = [,len(idx_durations)]
                    task_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,risk+2].long(), batch_y_train[:,1].float()) 
                    # weighted_task_loss = task_loss.data * case_weights[risk] 
                    loss_dict[task_name[risk]] = task_loss ## weighted_task_loss MOD
                    batch_loss += task_loss ##
                
                loss_dict['Loss'] = batch_loss ## sum of task losses
                # loss = batch_loss.clone() ## loss_dict[first_loss].clone(), Should be loss2

                # loss.backward()
                
                if batch_idx % 10 == 0:
                    batch_transference = {}
                    for combined_task in task_name: #self.loss_keys
                        batch_transference[combined_task] = {}
          
                    batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size] # MOD ["risk_{}".format(E)]
                    
                    for E in range(self.model.config.num_event): #self.loss_keys 
                        combined_task = task_name[E]
                        # optimizer.zero_grad() #?
                        preds = self.lookahead((batch_x_cat, batch_x_num), loss_dict[combined_task], optimizer, event=E) ##
                        
                        for i in range(len(task_name)):
                            c_name = task_name[i]
                            # if first_loss is None: first_loss = c_name
                            weights = case_weights[i] if self.weighting_tag else 1
                            # ->
                            nll_loss = criterion_fun(preds[1],  # phi
                                            batch_y_train[:,0].long(), # duration 
                                            batch_y_train[:,E + 2].long(), ## ?  should be i instead?
                                            batch_y_train[:,1].float()) # proportion
                            batch_transference[combined_task][c_name] = (
                                    (1.0 - (nll_loss/ (loss_dict[c_name] * weights))) / ## loss_dict standardized by case_weights
                                    optimizer.state_dict()['param_groups'][0]['lr']
                            ).detach().cpu().numpy()          
                    # self.optimizer.zero_grad() ##
                    # optimizer.zero_grad() # 0924
                
                    # Want to invert the dictionary so it's source_task => gradients on source task.
                    rev_transference = {source: {} for source in batch_transference}
                    for grad_task in batch_transference:
                        for source in batch_transference[grad_task]:
                            if 'Loss' in source:
                                continue
                            rev_transference[source][grad_task] = batch_transference[grad_task][source]
                
                    cumu_transference = copy.deepcopy(rev_transference)
                    for combined_task in task_name: # self.loss_keys
                        for recipient_task in task_name: # self.loss_keys
                            epoch_transference[combined_task][recipient_task] += ( #0801
                                    cumu_transference[combined_task][recipient_task] / ## batch_transference
                                    (num_train_batch / 10)) # len(self.train_loader)
                    
                for name, value in loss_dict.items():
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                ##
                
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            # Out of batch loop
            train_loss_list.append(epoch_loss / (batch_idx+1))
            
            # TAG
            train_stats = {}
            for name in task_name:
                meter = average_meters[name]
                train_stats[name] = meter.avg
       
            # return stats, epoch_transference
            # Save epoch-level transference metrics.
            for combined_task in task_name: ## self.loss_keys
                transference[combined_task].append(epoch_transference[combined_task])
    
            # Start validation phase: self.validate()
            if val_set is not None:
                average_meters = defaultdict(AverageMeter) ## TAG
                self.model.eval()
                val_loss = 0
                # for i in range(len(self.val_loader)): 
                # Standardize loss by case weights
                loss_dict = {}
                with torch.no_grad():
                    for risk in range(self.model.config.num_event): ## MOD 
                        phi_val = de_tuple(self.model.predict(tensor_val, val_batch_size, event=risk)) ##
                        val_task_loss = self.metrics[0](phi_val, 
                                                        tensor_y_val[:,0].long(), # ["risk_{}".format(risk)]
                                                        tensor_y_val[:,risk+2].long(), # ["risk_{}".format(risk)]
                                                        tensor_y_val[:,1].float()) # ["risk_{}".format(risk)]
                        # val_w_task_loss = val_task_loss *  case_weights[risk]
                        val_loss += val_task_loss ##
                        loss_dict[task_name[risk]] = val_task_loss # TAG
                        
                        
                    loss_dict['Loss'] = val_loss ## modified, sum of task losses each epoch (?) 
                if self.verbose: 
                    print("[Train-{}]: {}".format(epoch, epoch_loss / (batch_idx+1)))
                    print("[Val-{}]: {}".format(epoch, val_loss.item()))
                val_loss_list.append(val_loss.item())
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    print(f"early stops at epoch {epoch+1}")
                    # load best checkpoint
                    self.model.load_state_dict(torch.load(self.ckpt))
                    return train_loss_list, val_loss_list, (train_dict, valid_dict, transference)
                
                # TAG
                for name, value in loss_dict.items(): 
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                        
                val_stats = {}
                for name in task_name:
                    meter = average_meters[name]
                    val_stats[name] = meter.avg
                    
                ##
            else:
                print("[Train-{}]: {}".format(epoch, epoch_loss))
                
            # Train dict and val dict update (still inside epoch loop)
            for cname in task_name:
                train_dict[epoch][cname] = val_stats[cname] if self.train_by_val_stats else train_stats[cname] # Should alter val_stats instead?
                valid_dict[epoch][cname] = val_stats[cname]
                
            # if valid_total < best_val:
            #     best_val = valid_total
            
                # Omitted: Start test phase: self.test() 
    
            ##

        return train_loss_list, val_loss_list, (train_dict, valid_dict, transference)

    def fit(self, 
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):
        '''fit on the train_set, validate on val_set for early stop
        params should have the following terms:
        batch_size,
        epochs,
        optimizer,
        metric,
        '''
        if self.model.config.num_event == 1:
            return self.train_single_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    **kwargs,
            )
        
        elif self.model.config.num_event > 1:
            return self.train_multi_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    criterion_fun=self.metrics[0], # TAG
                    **kwargs,
            )
        
        else:
            raise ValueError
        
class AverageMeter(object): # TAG
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        # self.sumsq += float(val)**2
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std = np.std(self.lst)