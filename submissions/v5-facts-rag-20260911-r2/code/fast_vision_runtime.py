"""共享三视图ConvNeXt编码及固定分类头；每次调用重新前向，不读取历史预测。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def views(row):
    from PIL import Image,ImageOps
    with Image.open(row['path']) as im:image=ImageOps.exif_transpose(im).convert('RGB')
    w,h=image.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2
    return [image,ImageOps.mirror(image),image.crop((left,top,left+s,top+s))]

class FastVisionRuntime:
    def __init__(self,base,head,clock,*,batch_size=4,device='cuda'):
        import torch
        from transformers import AutoImageProcessor,AutoModel
        from cv_multiview_classifier import CVMultiviewClassifier
        if type(batch_size) is not int or not 1<=batch_size<=16:raise ValueError('视觉批量超出有界范围')
        self.torch=torch;self.device=torch.device(device);self.batch_size=batch_size;self.clock=clock
        torch.set_num_threads(2)
        base=Path(base);head=Path(head);self.base_sha=sha(base/'model.safetensors')
        metadata=json.loads((head/'manifest.json').read_text())
        self.classifier=CVMultiviewClassifier(head/'multiview_model.npz',head/'manifest.json',metadata['labels'],
                           {'base_sha256':self.base_sha,'frozen_base':True,'views':['full','horizontal_flip','center_square']})
        self.processor=AutoImageProcessor.from_pretrained(base,local_files_only=True)
        self.model=AutoModel.from_pretrained(base,local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.source={'base_sha256':self.base_sha,'config_sha256':sha(base/'config.json'),
                     'processor_sha256':sha(base/'preprocessor_config.json'),'head_sha256':self.classifier.model_sha256}

    def predict_many(self,rows):
        import numpy as np
        sync=(lambda:self.torch.cuda.synchronize(self.device)) if self.device.type=='cuda' else None
        with ThreadPoolExecutor(max_workers=4) as pool,self.torch.inference_mode():
            for start in range(0,len(rows),self.batch_size):
                batch=rows[start:start+self.batch_size]
                images=[im for group in pool.map(views,batch) for im in group]
                encoded=self.processor(images=images,return_tensors='pt').to(self.device)
                try:
                    with self.clock.model('convnext_three_views',batch_size=len(batch),model_sha256=self.base_sha,synchronize=sync):
                        outputs=self.model(**encoded)
                    features=outputs.pooler_output.float().cpu().numpy().reshape(len(batch),3,-1)
                    # 旧特征落盘为float16，保留量化行为以便检验分类一致性。
                    features=features.astype(np.float16).astype(np.float32)
                    with self.clock.model('cv_multiview_head',batch_size=len(batch),model_sha256=self.classifier.model_sha256):
                        predicted=[self.classifier.predict(f) for f in features]
                    for row,pred in zip(batch,predicted):yield row,pred
                finally:
                    for im in images:im.close()
                del encoded,outputs,features
