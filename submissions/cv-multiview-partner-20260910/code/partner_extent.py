"""由核验过的SAM落盘掩码和网格声明生成程度证据，物理尺度保持未知。"""
from pathlib import Path
import numpy as np
from report_contract import ContractError,read_json,digest_file,digest_json
from extent_evidence import extent_from_candidate


class MaskExtentPartner:
    def __init__(self,rows,mask_root):
        self.rows={r['sample_id']:r for r in rows};self.root=Path(mask_root)

    def __call__(self,arguments):
        if set(arguments)!={'sample_id'} or arguments['sample_id'] not in self.rows:raise ContractError('程度请求案件未登记')
        sid=arguments['sample_id'];row=self.rows[sid]
        if digest_file(row['path'])!=row['image_sha256']:raise ContractError('程度输入摘要错误')
        outputs={}
        for opt in ['adamw','sgd_momentum']:
            folder=self.root/opt;source=read_json(folder/'source.json');packet=read_json(folder/(sid+'.json'))
            if packet['sample_id']!=sid or packet['checkpoint_sha256']!=source['checkpoint_sha256']:
                raise ContractError('掩码来源或案件不一致')
            grid=source['mask_grid_hw']
            if not isinstance(grid,list) or len(grid)!=2 or any(type(v) is not int or v<=0 for v in grid):
                raise ContractError('掩码网格声明无效')
            maskfile=(folder/packet['mask_file']).resolve()
            if not maskfile.is_relative_to(folder.resolve()) or digest_file(maskfile)!=packet['mask_sha256']:
                raise ContractError('掩码文件越界或摘要变化')
            with np.load(maskfile,allow_pickle=False) as data:
                if 'masks' not in data:raise ContractError('掩码数组字段缺失')
                packed=data['masks']
            if packed.dtype!=np.uint8 or packed.ndim!=2:raise ContractError('掩码数组类型或维度无效')
            candidates=[]
            for raw in packet['candidates']:
                index=raw['mask_index']
                if type(index) is not int or not 0<=index<len(packed):raise ContractError('掩码索引无效')
                bits=np.unpackbits(packed[index])
                if bits.size!=grid[0]*grid[1] or int(bits.sum())!=raw['mask_grid_area_pixels']:
                    raise ContractError('掩码网格或像素数不一致')
                candidate={**raw,'mask_grid_size_hw':grid}
                candidate['extent']=extent_from_candidate(candidate,digest_json({'sample':sid,'opt':opt,'mask':packet['mask_sha256'],'index':index}))
                candidates.append(candidate)
            outputs[opt]={'checkpoint_sha256':source['checkpoint_sha256'],'preprocessing_sha256':digest_file(folder/'source.json'),'candidates':candidates}
        return {'status':'ok','result':{'sample_id':sid,'outputs':outputs},'model_calls':0,'new_visual_observation':False,
                'limitations':['候选掩码网格占比，不是病害确证、物理面积或构件受损比例；不同候选不可直接相加']}
