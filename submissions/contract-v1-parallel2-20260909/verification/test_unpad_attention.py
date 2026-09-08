"""合成张量独立对照：左补齐预填充与缓存解码必须等价于显式掩码注意力。"""
from pathlib import Path
import sys,unittest
from types import SimpleNamespace
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from unpad_attention import NAME,attention_forward,bind_padding,clear_padding

class UnpadAttentionTests(unittest.TestCase):
    def check_case(self,query_length,key_length,offsets):
        torch.manual_seed(73);torch.set_num_threads(2)
        b=len(offsets);q=torch.randn(b,6,query_length,8);k=torch.randn(b,2,key_length,8);v=torch.randn_like(k)
        module=SimpleNamespace(_c4_padding=(tuple(offsets),key_length if query_length>1 else key_length-2),is_causal=True)
        mask=torch.arange(key_length)[None,:]>=torch.tensor(offsets)[:,None]
        allowed=mask[:,None,None,:].expand(b,1,query_length,key_length).clone()
        if query_length>1:allowed &= torch.ones(query_length,key_length,dtype=torch.bool).tril()[None,None,:,:]
        reference=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,enable_gqa=True)
        actual,_=attention_forward(module,q,k,v,attention_mask=mask,scaling=8**-.5)
        actual=actual.transpose(1,2)
        for i,left in enumerate(offsets):
            qleft=left if query_length>1 else 0
            torch.testing.assert_close(actual[i,:,qleft:,:],reference[i,:,qleft:,:],atol=1e-6,rtol=1e-5)

    def test_left_padded_prefill_equals_masked_attention(self):self.check_case(11,11,[0,3,6])
    def test_cached_decode_equals_masked_attention(self):self.check_case(1,13,[0,3,6])
    def test_equal_length_batch_equals_masked_attention(self):self.check_case(11,11,[0,0])
    def test_invalid_padding_and_unbound_batch_are_rejected(self):
        module=SimpleNamespace(config=SimpleNamespace(_attn_implementation=NAME),num_key_value_groups=3)
        model=SimpleNamespace(modules=lambda:[module])
        with self.assertRaisesRegex(ValueError,'连续左补齐'):bind_padding(model,torch.tensor([[1,0,1]]))
        found=bind_padding(model,torch.tensor([[0,1,1]]));self.assertEqual(module._c4_padding,((1,),3))
        clear_padding(found);q=torch.zeros(1,6,3,8);k=torch.zeros(1,2,3,8)
        with self.assertRaisesRegex(ValueError,'未绑定'):attention_forward(module,q,k,k)

if __name__=='__main__':unittest.main()
