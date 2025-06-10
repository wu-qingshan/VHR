import math
import torch
import numpy as np
import transformers
import torch.nn.functional as F
from functools import partial
from transformers.models.llama.modeling_llama import *
import random

from utils import get_layers


class EnHancingAttentionHead:
    def __init__(self, beta, img_start, img_end):
        self.beta = beta 
        self.s = img_start 
        self.e = img_end 
        self.hist_attn_weights = None 
        self.max_head_idx = None

    def compute_dense_head(self, attention_matrix):
        k = random.randint(self.s, self.e)
        # k = 0
        # print('k value: ', k)
        B, H, N, _ = attention_matrix.shape
        r = N
        M = 1-torch.eye(N, device=attention_matrix.device).unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
        img_token_masked = (attention_matrix * M)[:, :, k:r, :]  # [B, H, r-k, N]

        avg_attention = img_token_masked.mean(dim=2) 

        vision_sink_indices = avg_attention > self.beta

        vision_sink_matrix = vision_sink_indices.sum(dim=-1)

        return vision_sink_matrix

    def enhancing_attention(self, attention_matrix):
        vision_sink_matrix = self.compute_dense_head(attention_matrix)
        # print(f"vision_sink_matrix: {vision_sink_matrix}") 
        # 如果 vision_sink_matrix 均为空，直接返回原始的 attention_matrix_enhanced
        if torch.all(vision_sink_matrix == 0):
            print(f'=============================================')
            print(f'Layer 2 no copies')
            return attention_matrix, None
        
        attention_matrix_enhanced = torch.zeros_like(attention_matrix)
        
        for batch in range(attention_matrix.size(0)): 
            H = []  # 用于存储 (Ci,j, j) 的列表
            
            for head in range(attention_matrix.size(1)): 
                vision_sinks = vision_sink_matrix[batch, head]
                
                H.append((vision_sinks.item(), head))
            
            _, max_dense_head_idx = max(H, key=lambda x: x[0])
            
            for head in range(attention_matrix.size(1)):
                attention_matrix_enhanced[batch, head] = attention_matrix[batch, max_dense_head_idx].clone()

        return attention_matrix_enhanced, max_dense_head_idx


def enhanced_attn_forward(
        self,
        eah: EnHancingAttentionHead,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
)-> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
        
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # print("query_states shape:", query_states.shape)
        # print("key_states shape:", key_states.shape)
        # print("value_states shape:", value_states.shape)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # print("query_states shape:", query_states.shape)
    # print("key_states shape:", key_states.shape)
    # print("value_states shape:", value_states.shape)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
    if attn_weights.shape[-2] > 1:  #prompt阶段
        attn_weights, eah.max_head_idx = eah.enhancing_attention(attn_weights)
        # print(f"Max dense head index is head {eah.max_head_idx}")
    
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)    
    
    attn_output = torch.matmul(attn_weights.to(query_states.dtype), value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def replace_attention_with_eah(model, beta, img_start, img_end, layers_to_modify):
    eah = EnHancingAttentionHead(beta, img_start, img_end)
    
    for idx, layer in enumerate(get_layers(model)):
        if idx in layers_to_modify:
            layer.self_attn.forward = partial(enhanced_attn_forward, layer.self_attn, eah)
            
