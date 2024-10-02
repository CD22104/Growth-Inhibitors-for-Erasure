from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionPipeline
import numpy as np
import torch
import abc
import generate_utils as g_utils
import seq_aligner

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        raise NotImplementedError


    def __call__(self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[:h // 2] = self.forward(attn[:h // 2], is_cross, place_in_unet,erase_attn[:h // 2])

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn    
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        return attn
     
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, erase_attn):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2: 
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {}
        for key in self.attention_store:
            attention_list = self.attention_store[key]
            average_attention[key] = []
            for item in attention_list:
                average_attention[key].append(item / self.cur_step)
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace,erase_attn):
        raise NotImplementedError
         
    def forward(self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet,erase_attn)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size) 
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:]) 
            erase_attn = erase_attn.reshape(self.batch_size, h, *erase_attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            erase_attn_base=erase_attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce,erase_attn_base) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, erase_targets, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],tokenizer,device):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = g_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float: 
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

class AttentionInhibit(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace, erase_attn):
        erase_attn_supress = erase_attn[:, :, :, self.erase_mapper[0]] * self.equalizer[:, None, None, :]
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3) 
        attn_replace = attn_base_replace * self.alphas + erase_attn_supress * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, erase_targets,  num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, tokenizer, device):
        super(AttentionInhibit, self).__init__(prompts, erase_targets, num_steps, cross_replace_steps, self_replace_steps,tokenizer,device)
        self.equalizer = equalizer.to(device)
        self.mapper, erase_mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.erase_mapper = erase_mapper.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionErase(AttentionInhibit):
    def forward(self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet,erase_attn)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size) 
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:]) 
            erase_attn = erase_attn.reshape(self.batch_size, h, *erase_attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            erase_attn_base=erase_attn[1:]
            if is_cross:
                self.cross_step.append(erase_attn_base[:,:,:,1:self.extract-1])
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce,erase_attn_base) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str,erase_attn):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[:h // 2] = self.forward(attn[:h // 2], is_cross, place_in_unet,erase_attn[:h // 2])
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.data.append(self.cross_step)
            self.cross_step=[]
            self.between_steps()
        return attn  

    def __init__(self, prompts, erase_targets,  num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, tokenizer, device):
        super(AttentionErase, self).__init__(prompts, erase_targets, num_steps, cross_replace_steps, self_replace_steps, equalizer, tokenizer, device)
        self.cross_step=[]
        self.data=[]
        join_result = " ".join(erase_targets)
        tokens_prompt = tokenizer.encode(join_result)
        self.extract = len(tokens_prompt)


