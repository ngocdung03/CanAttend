import json
import torch
torch.set_num_threads(10)
torch.set_num_interop_threads(10)

from collections import defaultdict
import matplotlib.pyplot as plt

from canattend.modeling.evaluate_utils import Evaluator
from canattend.utils import set_random_seed
from canattend.modeling.model import CanAttend, Survtrace
from canattend.modeling.train_utils import Trainer
from canattend.dataset import load_data
from canattend.helper import average_dict
from canattend.plot_helper import attention_heatmap, get_hp, CauseSpecificNet, de_encode, calculate_hazard, plot_roc_combo, plot_roc

import os, sys
import re
import pandas as pd
import copy
import numpy as np
import torch
from datetime import datetime
from canattend.config import caConfig
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder, StandardScaler #KBinsDiscretizer,
import sklearn.metrics as metrics
from canattend.utils import LabelTransform
import pickle 

os.chdir('..')
from surv0913.lib.preprocessing import get_cancers, get_features, imputation
from surv0913.run_surv import dat_process
os.chdir('canattend')

# Create args object
EXTRA = '' #Train-aftertune
CUTS = [5, 10, 15] ## 12
SAMPLE = None
GROUP_LIST = [[0, 1, 5, 6, 7, 8, 9],
			  [2, 3, 4, 5, 6, 7, 8, 9],
			  [i for i in range(10)]]
DISCRETE = True
SPARSE = True
WEIGHT = False
N_EPOCHS = 20
DURATION = 10
N_WORKER = 0
RETRIEVING_TAG = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class args():
	def __init__(self):
		super(args, self).__init__()
		self.dataset = '../surv0913/dataset/ba8.csv' #yonsei_final_moa
		self.cancer = '../surv0913/dataset/cancers_new.txt'
		self.feature = '../surv0913/dataset/features.txt'
		self.test_rate = 0.2
		self.imputation = 'mean'
		self.by_sex = True
		self.tasks =  ['THROI', 'STOMA', 'BREAC', 'CRC', 'LUNG', 'PROST', 'LIVER', 'KIDNE', 'UTE_CER', 'LYMPH'] 
		self.new = True
		self.version = 2
		self.drop_normal = False
		self.num_durations = DURATION
		self.n_workers = N_WORKER
		self.device = device
		
ARGS = args()      
cancer, cancer_time = get_cancers(ARGS.cancer)
feature0 = get_features(ARGS.feature)[1:] 
INP = len(feature0)

# Tuned hyperparameters
def get_hp(filename):
	with open(filename, 'r') as f:
		return [x.strip('\n') for x in f.readlines() if '#' not in x]
	
hp_list = get_hp('./implement_checkpoints/tuned_hparams_tag_ysdat.txt')

for GROUP in range(len(GROUP_LIST)):
    # Specified config
	caConfig = EasyDict(
		{
			'data': 'kcpsii', 
			'num_durations': 5, 
			'horizons': CUTS, 
			'seed': 1234,
			'checkpoint': './checkpoints/canattend.pt',
			'vocab_size': 8, 
			'hidden_act': 'gelu',
			'attention_probs_dropout_prob': 0.1,
			'early_stop_patience': 5,
			'initializer_range': 0.02, #0.001
			'layer_norm_eps': 1e-12,
			'discrete_time': DISCRETE,
			'discrete_time_test': False,
			'task': [ARGS.tasks[i] for i in GROUP_LIST[GROUP]]
		}
	)

	set_random_seed(caConfig['seed'])
	now = datetime.now()
	current_time = now.strftime("%y-%m-%d_%H.%M.%S")

	MOD_NAME = EXTRA + f'{current_time}_{GROUP_LIST[GROUP]}_sparse{SPARSE}_cuts{CUTS}'
	caConfig['model_name'] = MOD_NAME
 
	# Config model based on hparams
	tuned_hp = json.loads(hp_list[GROUP])
	ATT_DROP = tuned_hp['dropout']
	LR = tuned_hp['learning_rate'] #0.00040474061064163703
	BS = tuned_hp['batch_size'] #64

	hidden_size = tuned_hp['hidden_size']
	intermediate_size = tuned_hp['intermediate_size']
	num_hidden_layers = tuned_hp['num_hidden_layers'] 
	num_attention_heads = tuned_hp['num_attention_heads']

	hparams = {
		'batch_size': BS,
		'weight_decay': 1e-4,
		'learning_rate': LR,
		'epochs': N_EPOCHS,
	}

	caConfig['attention_probs_dropout_prob'] = ATT_DROP
	caConfig['hidden_size'] = hidden_size
	caConfig['intermediate_size'] = intermediate_size
	caConfig['num_hidden_layers'] = num_hidden_layers
	caConfig['num_attention_heads'] = num_attention_heads
 
	# Load data
	df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(caConfig, ARGS, sample=SAMPLE)

	# Training
	# get model
	model = CanAttend(caConfig, new_ver=SPARSE) ##

	# initialize a trainer
	trainer = Trainer(model, weighting_tag=WEIGHT, device=ARGS.device) #Trainer2
	train_loss, val_loss, tag_outputs = trainer.fit((df_train, df_y_train), (df_val, df_y_val), ##
			batch_size=hparams['batch_size'],
			epochs=hparams['epochs'],
			learning_rate=hparams['learning_rate'],
			weight_decay=hparams['weight_decay'],)

	# If TAG matrix needed to be exported
	if RETRIEVING_TAG:
		train_dict, valid_dict, transference = tag_outputs
		revised_integrals = average_dict(transference)
		print('Average transference', revised_integrals)
  
	# Save model
	path = f'./implement_checkpoints/{caConfig["model_name"]}.pt'
	torch.save(model.state_dict(), path)

	# Evaluation
	evaluator = Evaluator(df, df_train.index)
	metric_dict = evaluator.eval(model, (df_test, df_y_test), val_batch_size=None, allow_all_censored=True)
 
	# Log IPCW c-index
	c_indices = []
	for key in metric_dict.keys():
		if 'ipcw' in key:
			c_indices.append(metric_dict[key])
   
	task = model.config['task']
	now = datetime.now()
	current_time = now.strftime("%y-%m-%d_%H.%M.%S")
	with open('./implement_checkpoints/c_ind_kcpsii.txt', 'a') as f:
		f.write('\n')
		f.write(f"{caConfig['model_name']}: \n{metric_dict}")
		f.write('\n')
		for i in range(len(c_indices)):
			f.write(f'{c_indices[i]},')
			if (i + 1) % 3 == 0:
				f.write('\n')

	# Plot training and validation loss
	plt.plot(train_loss, label='train')
	plt.plot(val_loss, label='val')
	plt.legend(fontsize=20)
	plt.xlabel('epoch',fontsize=20)
	plt.ylabel('loss', fontsize=20)
	# plt.show()
	plt.savefig(f'.implement_checkpoints/plot/loss_{caConfig['model_name']}.png', dpi=300, bbox_inches='tight')
	# Close the plot to free up memory
	plt.close()
