"""验证图像骨干复用的图像边界、可变列表隔离与异常恢复。"""
from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from infer_grounding_holdout import reuse_image_backbone

class Backbone(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,pixels,mask):
        self.calls+=1
        return [(pixels+1,mask)],[pixels+2]

class RegionReuseTests(unittest.TestCase):
    def test_one_image_computed_once_and_lists_are_isolated(self):
        module=Backbone().eval();pixels=torch.zeros(1,3,2,2);mask=torch.ones(1,2,2)
        with torch.inference_mode():
            with reuse_image_backbone(module,pixels,mask):
                first=module(pixels,mask);first[1].append(torch.zeros(1))
                second=module(pixels,mask)
                self.assertEqual(len(second[1]),1)
                self.assertEqual(module.calls,1)
                torch.testing.assert_close(second[0][0][0],pixels+1)
            module(pixels,mask)
        self.assertEqual(module.calls,2)

    def test_wrong_image_rejected_and_original_forward_restored(self):
        module=Backbone().eval();pixels=torch.zeros(1,3,2,2);mask=torch.ones(1,2,2)
        original=module.forward
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'不得跨图像'):
                with reuse_image_backbone(module,pixels,mask):module(pixels.clone(),mask)
            self.assertEqual(module.forward,original)
        with self.assertRaisesRegex(ValueError,'冻结推理'):
            with reuse_image_backbone(module,pixels,mask):pass

if __name__=='__main__':unittest.main()
