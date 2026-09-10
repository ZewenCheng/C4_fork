"""供新输入使用的完整落盘区域适配，绑定当前清单和检测权重。"""
from pathlib import Path
import math
from report_contract import ContractError,read_json,digest_file


class SavedRegions:
    def __init__(self,rows,online,checkpoint_root):
        self.rows={r['sample_id']:r for r in rows};self.online=Path(online);self.checkpoints=Path(checkpoint_root)
        if len(self.rows)!=len(rows) or any(not isinstance(s,str) or Path(s).name!=s or s in ('.','..') for s in self.rows):
            raise ContractError('区域清单案件重复或标识无效')
        self.weights={opt:digest_file(self.checkpoints/('rtdetr_'+opt)/'last.pt') for opt in ['adamw','sgd_momentum']}
        self.configs={opt:read_json(self.checkpoints/('rtdetr_'+opt)/'configuration.json') for opt in self.weights}

    def __call__(self,arguments):
        if set(arguments)!={'sample_id'} or arguments['sample_id'] not in self.rows:raise ContractError('区域请求案件未登记')
        sid=arguments['sample_id'];row=self.rows[sid]
        if digest_file(row['path'])!=row['image_sha256']:raise ContractError('区域缓存输入内容变化')
        outputs={}
        for opt in self.weights:
            folder=self.online/'rtdetr_holdout'/opt
            data=read_json(folder/(sid+'.json'));receipt=read_json(folder/'complete.json')
            if data['sample_id']!=sid or data['checkpoint_sha256']!=self.weights[opt] or receipt['checkpoint_sha256']!=self.weights[opt]:
                raise ContractError('区域缓存身份或权重不符')
            config=self.configs[opt]
            if data.get('base_manifest_sha256')!=config['base_manifest_sha256']:raise ContractError('区域基底版本不符')
            if data.get('image_sha256',row['image_sha256'])!=row['image_sha256']:raise ContractError('区域缓存图像版本不符')
            for candidate in data['candidates']:
                box=candidate['box_normalized_xyxy'];score=candidate['uncalibrated_score']
                if (len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in box)
                    or box[0]>box[2] or box[1]>box[3] or type(score) not in (int,float)
                    or not math.isfinite(score) or not 0<=score<=1 or candidate['query'] not in config['query_vocabulary']):
                    raise ContractError('区域候选坐标、分数或词表无效')
            outputs[opt]=data
        return {'status':'ok','result':{'sample_id':sid,'outputs':outputs},'cache_mode':'full_saved_top20',
                'model_calls':0,'new_visual_observation':False,'limitations':['前20条落盘池，不是全部内部查询']}
