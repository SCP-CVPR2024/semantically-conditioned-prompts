import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import torch.nn.functional as F
import math  

class QKV_ps(nn.Module):
    def __init__(self, d_model, k=16, pool_size=64, fixed_prompts=0, return_att = False):
        super().__init__()
        self.d_model=d_model
        self.k = k
        assert d_model % k == 0
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.bias_Q = nn.Parameter(torch.empty(d_model))
        self.promptPool_K = nn.Parameter(torch.empty(pool_size, d_model//k)) 
        self.promptPool_V = nn.Parameter(torch.empty(pool_size, d_model))
        self.dropout = nn.Dropout(0.1) 
        self.layerNorm = nn.LayerNorm(d_model)
        self.return_att = return_att
        
        if fixed_prompts != 0:
            self.fixed_prompts = nn.Parameter(torch.empty(fixed_prompts, d_model))
            init.kaiming_normal_(self.fixed_prompts , mode='fan_in', nonlinearity='relu')
        else:
            self.fixed_prompts = None
        # init.kaiming_normal_(self.W_Q , mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.promptPool_K , mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.promptPool_V , mode='fan_in', nonlinearity='relu')
    def forward(self, 
                CLS, # B # 1 # E
                pool_idxs):
        # modality agnostic
        batch_size = CLS.shape[0]
        W_Q = self.W_Q # B # P # E    
        bias = self.bias_Q # E  
        prompts_k = self.promptPool_K # B # P # E
        prompts_v = self.promptPool_V # B # P # E
        # generate queries
        CLS_norm = self.layerNorm(CLS)
        query = torch.matmul(CLS_norm, W_Q) 
        query = query + bias
        # add bias
        query = query.view(query.size(0),self.k,-1)
        similarity = torch.matmul(query, prompts_k.transpose(-2,-1))
        div = math.sqrt(self.d_model) # //self.k
        similarity = similarity/div
        A = F.softmax(similarity, dim=-1)
        # attention bias
        A = self.dropout(A)
        selected_prompts = torch.matmul(A, prompts_v)
        if self.fixed_prompts != None:
            fixed_prompts = self.fixed_prompts.repeat(batch_size,1,1)
            selected_prompts = torch.cat([fixed_prompts, selected_prompts], dim=1)
        if self.return_att:
            return selected_prompts, A    
        else:
            return selected_prompts    
    
class add_agnostic_prompts(nn.Module):
    def __init__(self, d_model, pool_size=64):
        super().__init__()
        self.promptPool = nn.Parameter(torch.empty(pool_size, d_model))
        init.kaiming_normal_(self.promptPool , mode='fan_in', nonlinearity='relu')
    def forward(self, 
                x, # B # 1 # E
                ):
        # modality agnostic
        batch_size = x.shape[0]
        promptPool = self.promptPool.repeat(batch_size,1,1)
        return promptPool        