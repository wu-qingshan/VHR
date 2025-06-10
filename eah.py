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
        """
        初始化类：包含阈值 beta 和图像 token 的起始、结束索引
        """
        self.beta = beta  # 设置阈值 beta
        self.s = img_start  # 图像 token 的起始索引
        self.e = img_end  # 图像 token 的结束索引
        self.hist_attn_weights = None  # 初始化历史注意力权重矩阵
        self.max_head_idx = None

    def compute_dense_head(self, attention_matrix):
        """
        计算每个头 (head) 的视觉汇集数量矩阵
        """
        k = random.randint(self.s, self.e)
        # k = 0
        # print('k value: ', k)
        # 生成掩码矩阵 M（对角线为 0）
        B, H, N, _ = attention_matrix.shape
        r = N
        M = 1-torch.eye(N, device=attention_matrix.device).unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
        # 提取图像 token 的注意力分数
        img_token_masked = (attention_matrix * M)[:, :, k:r, :]  # [B, H, r-k, N]

        avg_attention = img_token_masked.mean(dim=2)  # 对行求和

        # 获取满足阈值的视觉汇集位置 (x, y)
        vision_sink_indices = avg_attention > self.beta

        # 统计每个头的视觉汇集数量
        vision_sink_matrix = vision_sink_indices.sum(dim=-1)

        return vision_sink_matrix

    def enhancing_attention(self, attention_matrix):
        """
        根据视觉汇集数量增强注意力矩阵
        """
        # 计算视觉汇集矩阵
        vision_sink_matrix = self.compute_dense_head(attention_matrix)
        # print(f"vision_sink_matrix: {vision_sink_matrix}")  # 期望：[B, H, N]
        # 如果 vision_sink_matrix 均为空，直接返回原始的 attention_matrix_enhanced
        if torch.all(vision_sink_matrix == 0):
            print(f'=============================================')
            print(f'Layer 2 no copies')
            return attention_matrix, None
        
        # 初始化增强后的注意力矩阵
        attention_matrix_enhanced = torch.zeros_like(attention_matrix)
        
        # 遍历每个 batch 进行处理
        for batch in range(attention_matrix.size(0)):  # batch 维度
            H = []  # 用于存储 (Ci,j, j) 的列表
            
            # 遍历每个头（head）
            for head in range(attention_matrix.size(1)):  # head 维度
                # 获取当前头的视觉汇集数量
                vision_sinks = vision_sink_matrix[batch, head]
                
                # 将视觉汇集数量及对应的头索引添加到 H 中
                H.append((vision_sinks.item(), head))
            
            # 找到视觉汇集数量最多的头
            _, max_dense_head_idx = max(H, key=lambda x: x[0])
            
            # 遍历所有头并更新头矩阵
            for head in range(attention_matrix.size(1)):
                attention_matrix_enhanced[batch, head] = attention_matrix[batch, max_dense_head_idx].clone()
            # print(f'Layer 2 copy head{max_dense_head_idx}')
            # print(H)

        return attention_matrix_enhanced, max_dense_head_idx
        # return attention_matrix_enhanced
    
    

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
    # '''
    # 思路1: 全局EAH
    # '''
    # if attn_weights.shape[-2] == 1:  # 推理生成阶段
    #     zero_column = torch.zeros((eah.hist_attn_weights.shape[0], eah.hist_attn_weights.shape[1], eah.hist_attn_weights.shape[2], 1), device=eah.hist_attn_weights.device)
    #     eah.hist_attn_weights = torch.cat((eah.hist_attn_weights, zero_column), dim=3)  # 在历史 Attention Weights 的最前面增加一列0
    #     eah.hist_attn_weights = torch.cat((eah.hist_attn_weights, attn_weights), dim=2)
    # else:  # prompt阶段
    #     eah.hist_attn_weights = attn_weights.clone()
        
    # if attn_weights.shape[-1] > 700:
    #     pass
    # # 添加 EAH 部分
    # if attn_weights.shape[-2] > 1:  # prompt阶段
    #     attn_weights = eah.enhancing_attention(eah.hist_attn_weights)
    # else:  # 生成推理阶段
    #     attn_weights = eah.enhancing_attention(eah.hist_attn_weights)[:,:,-1,:].unsqueeze(-2)

    # print("attn_weights shape:", attn_weights.shape)
    # '''
    # 思路2: prompt阶段EAH+广播
    # '''
    # if attn_weights.shape[-2] > 1:  #prompt阶段
    #     attn_weights, eah.max_head_idx = eah.enhancing_attention(attn_weights)
    # else:  # 生成推理阶段
    #     for batch in range(attn_weights.shape[0]):
    #         for head in range(attn_weights.shape[1]):
    #             attn_weights[batch, head] = attn_weights[batch, eah.max_head_idx].clone()
                
    '''
    思路3: 仅对prompt阶段进行EAH
    '''
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

'''
替换模型原理：
在Python中,函数是一等公民，可以像普通变量一样被赋值和替换。
例如: 替换带掉类module中的forward方法就可以使用: 
def new_forward(self, x):
    return x + 10
model.forward = new_forward.__get__(model, model.__class__)  # 绑定到实例:替换函数.__get__(目标实例, 类) model.__class__表示实例meodel所在的类)
'''
def replace_attention_with_eah(model, beta, img_start, img_end, layers_to_modify):
    # 初始化 EAH 模块
    eah = EnHancingAttentionHead(beta, img_start, img_end)
    
    for idx, layer in enumerate(get_layers(model)):
        if idx in layers_to_modify:
            layer.self_attn.forward = partial(enhanced_attn_forward, layer.self_attn, eah)
            

if __name__ == "__main__":
    # 替换是否成功测试
    from utils import *

    def test_get_model():
        # 设置设备（CUDA或CPU）
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        
        # 模型路径
        model_path = "/data/huazhenglin/model_cache/models--llava-hf--llava-1.5-7b-hf"

        # 调用get_model函数
        processor, model, template, ans_start, num_img_tokens, image_start = get_model(model_path, device)

        # 输出模型信息，验证是否加载正确
        print(f"Processor: {processor}")
        print(f"Model: {model}")
        print(f"Template: {template}")
        print(f"Answer start token: {ans_start}")
        print(f"Number of image tokens: {num_img_tokens}")
        print(f"Image start token position: {image_start}")

        # 验证模型的注意力层是否替换成功
        attention_layer_replaced = False
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                # 检查替换后的forward方法
                if hasattr(module, 'forward'):
                    attention_layer_replaced = True
                    print(f"Successfully replaced forward method in module: {name}")
                else:
                    print(f"Forward method not replaced in module: {name}")

        # 如果没有找到替换的注意力层，抛出异常
        if not attention_layer_replaced:
            print("Failed to replace the attention layer.")
            return
    
    test_get_model()
    
    # EAH算法测试
    # B, H, N = 1, 4, 3
    # attention_matrix = torch.tensor([[[[0.7314, 0.1738, 0.6979],
    #                                 [0.2087, 0.2514, 0.6493],
    #                                 [0.5961, 0.5475, 0.3812]],

    #                                 [[0.9215, 0.8532, 0.2492],
    #                                 [0.6884, 0.2644, 0.9124],
    #                                 [0.5978, 0.7390, 0.1311]],

    #                                 [[0.7547, 0.9125, 0.6518],
    #                                 [0.5122, 0.3441, 0.9679],
    #                                 [0.2093, 0.8595, 0.9524]],

    #                                 [[0.4322, 0.9295, 0.6971],
    #                                 [0.5690, 0.3162, 0.0731],
    #                                 [0.8660, 0.0930, 0.7360]]]]).float()
    # beta = 1  # 设置阈值
    # img_start = 0  # 图像token的起始索引
    # img_end = N-1   # 图像token的结束索引

    # # 实例化EnHancingAttentionHead类
    # enhancer = EnHancingAttentionHead(beta, img_start, img_end)

    # # 测试compute_dense_head
    # vision_sink_matrix = enhancer.compute_dense_head(attention_matrix)

    # # 测试enhancing_attention
    # enhanced_attention = enhancer.enhancing_attention(attention_matrix)
    # print(f"Enhanced attention matrix: {enhanced_attention}")  # 期望：[B, H, N, N]




