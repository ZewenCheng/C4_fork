"""长文本批量注意力：去掉每条序列的左补齐，复用平台已验证的无掩码SDPA。"""
import torch
from torch.nn import functional as F

NAME='c4_unpadded_sdpa'

def register_backend():
    from transformers import AttentionInterface,AttentionMaskInterface
    from transformers.masking_utils import flash_attention_mask
    AttentionInterface.register(NAME,attention_forward)
    # 仅复用2D补齐掩码格式，不调用flash-attn内核。
    AttentionMaskInterface.register(NAME,flash_attention_mask)

def bind_padding(model,mask):
    if mask.ndim!=2 or mask.device.type!='cpu':raise ValueError('补齐身份必须来自当前CPU分词结果')
    batch,width=mask.shape
    counts=mask.sum(-1).tolist();offsets=tuple(width-int(n) for n in counts)
    expected=torch.arange(width)[None,:]>=torch.tensor(offsets)[:,None]
    if any(n<=0 for n in counts) or not torch.equal(mask,expected.to(mask.dtype)):
        raise ValueError('只接受非空序列的连续左补齐掩码')
    found=[]
    for module in model.modules():
        if getattr(getattr(module,'config',None),'_attn_implementation',None)==NAME and hasattr(module,'num_key_value_groups'):
            module._c4_padding=(offsets,width);found.append(module)
    if not found:raise ValueError('模型缺少本轮配置的文本注意力模块')
    return found

def clear_padding(modules):
    for module in modules:module._c4_padding=None

def attention_forward(module,query,key,value,attention_mask=None,dropout=0.0,scaling=None,is_causal=None,**kwargs):
    padding=getattr(module,'_c4_padding',None)
    if padding is None:raise ValueError('未绑定当前批量的补齐身份，拒绝计算')
    offsets,initial_width=padding
    batch,_,query_length,_=query.shape;key_length=key.shape[2]
    if len(offsets)!=batch or key.shape[0]!=batch or value.shape[0]!=batch or key_length<initial_width:
        raise ValueError('注意力批量、缓存与当前输入不一致')
    if query_length not in (1,key_length) or dropout!=0 or not getattr(module,'is_causal',True):
        raise ValueError('该后端仅用于当前完整预填充及单词元贪心解码')
    if attention_mask is not None and (attention_mask.ndim!=2 or attention_mask.shape!=(batch,key_length)):
        raise ValueError('注意力补齐掩码形状异常')
    def attend(q,k,v):
        return F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.0,scale=scaling,
                                             is_causal=query_length>1,enable_gqa=True)
    if not any(offsets):
        result=attend(query,key,value)
    else:
        outputs=[]
        for i,left in enumerate(offsets):
            qleft=left if query_length>1 else 0
            out=attend(query[i:i+1,:,qleft:,:],key[i:i+1,:,left:,:],value[i:i+1,:,left:,:])
            if qleft:out=F.pad(out,(0,0,qleft,0))
            outputs.append(out)
        result=torch.cat(outputs,dim=0)
    return result.transpose(1,2).contiguous(),None
