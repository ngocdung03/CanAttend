from typing import Sequence
import torch
from torch import nn
import numpy as np
from pycox.models import utils
import pandas as pd
import torchtuples as tt
import pdb
import torch.nn.functional as F
from .modeling_bert import BaseModel, BertEmbeddings, BertEncoder, BertCLS, BertCLSMulti 
from ..utils import pad_col, de_tuple
from ..config import STConfig
 
class CanAttend(BaseModel):

    def __init__(self, config: STConfig): #new_ver: bool = True
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config)
        print("Using modified version of BertEncoder.py")
            
        self.cls = BertCLSMulti(config)
        self.config = config
        self.init_weights()
        self.duration_index = config['duration_index']
        self.use_gpu = False

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        event=0, # output the prediction for different competing events
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(embedding_output) #, output_attentions=output_attentions
        # if self.config.output_attentions:

        ###
        sequence_output = encoder_outputs[0]
        
        predict_logits = self.cls(sequence_output, event=event)
        return sequence_output, predict_logits, encoder_outputs[2] if len(encoder_outputs) > 2 else sequence_output, predict_logits ##1024 
        # return sequence_output, predict_logits 

    def predict(self, x_input, batch_size=None, event=0):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, :self.config.num_categorical_feature]
            x_input_num = x_input.iloc[:, self.config.num_categorical_feature:]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).long()
        else:
            x_cat = x_input[:, :self.config.num_categorical_feature].long()
            x_num = x_input[:, self.config.num_categorical_feature:].float()
        
        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:
                preds0 = self.forward(x_cat, x_num, event=event)
                
                if len(preds0) > 2: ##
                    preds = (preds0[1], preds0[2])
                else:
                    preds = preds0[1]
 
            else: # this condition will not return attn matrix
                preds = []
                attn = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx*batch_size:(idx+1)*batch_size]
                    batch_x_cat = x_cat[idx*batch_size:(idx+1)*batch_size]
                    batch_pred = self.forward(batch_x_cat, batch_x_num, event=event)
                    preds.append(batch_pred[1])
                    if len(batch_pred) > 2: #batch_pred[2] available
                        attn.append(batch_pred[2])
                preds0 = torch.cat(preds)
                attn = torch.cat(attn)
                
                if len(batch_pred) > 2: ##
                    preds = (preds0, attn)
                    
                else:
                    preds = preds0
                
        ## 1024
        # Check if result is a tuple of two tensors
        if isinstance(preds, tuple) and len(preds) == 2:
            # Return the result as a tuple
            return preds
        elif isinstance(preds, torch.Tensor):
            # Return the result directly if it's a single tensor
            return preds

    def predict_hazard(self, input_ids, batch_size=None, event=0):
        preds0 = self.predict(input_ids, batch_size, event=event)
        preds = de_tuple(preds0) # extract the output of self.predict() regardless output_attentions is True or False

        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard ## hazard, preds0[1] if len()>1
    
    def predict_risk(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return 1 - surv

    def predict_surv(self, input_ids, batch_size=None, event=0):
        hazard = self.predict_hazard(input_ids, batch_size, event=event)
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv
    
    def predict_surv_df(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)


class Survtrace(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLS(config)
        self.config = config
        self.init_weights()
        self.duration_index = config['duration_index']
        self.use_gpu = False

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[1]
        
        # do pooling
        # sequence_output = (encoder_outputs[1][-2] + encoder_outputs[1][-1]).mean(dim=1)

        predict_logits = self.cls(encoder_outputs[0])

        return sequence_output, predict_logits

    def predict(self, x_input, batch_size=None):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, :self.config.num_categorical_feature]
            x_input_num = x_input.iloc[:, self.config.num_categorical_feature:]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).long()
        else:
            x_cat = x_input[:, :self.config.num_categorical_feature].long()
            x_num = x_input[:, self.config.num_categorical_feature:].float()
        
        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:

                    preds = self.forward(x_cat, x_num)[1]
            else:
                preds = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx*batch_size:(idx+1)*batch_size]
                    batch_x_cat = x_cat[idx*batch_size:(idx+1)*batch_size]
                    batch_pred = self.forward(batch_x_cat,batch_x_num)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None):
        preds = self.predict(input_ids, batch_size)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard
    
    def predict_risk(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return 1- surv

    def predict_surv(self, input_ids, batch_size=None, epsilon=1e-7):
        hazard = self.predict_hazard(input_ids, batch_size)
        # surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv
    
    def predict_surv_df(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)