"""下载政府公开规范并逐页OCR，保留版面坐标；OCR不直接升格为评级规则。"""
import sys,subprocess,importlib,json,os,time
from pathlib import Path
from download_assets import ROOT,dump,sha
deps=ROOT/'deps';sys.path.insert(0,str(deps))
packages=['pymupdf==1.26.4','rapidocr_onnxruntime==1.4.4','onnxruntime==1.22.1','pyclipper==1.3.0.post6','shapely==2.1.1','coloredlogs==15.0.1','humanfriendly==10.0','flatbuffers==25.2.10']
try:import fitz;from rapidocr_onnxruntime import RapidOCR
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','--target',str(deps),'--no-deps','--disable-pip-version-check',*packages],check=True)
    importlib.invalidate_caches()
    import fitz
    from rapidocr_onnxruntime import RapidOCR
import requests
import numpy as np

SOURCES={
 'JTG_T_H21_2011': 'https://xxgk.mot.gov.cn/2020/jigou/glj/202006/P020240521541427295157.pdf',
 'CJJ_99_2017': 'https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud.zj.gov.cn/jcms_files/jcms1/web3613/site/attach/0/1b4928f48d5d4ed29c7c671cc9adfd39.pdf',
}
def main():
    out=ROOT/'standards';out.mkdir(exist_ok=True)
    engine=None;results={}
    for name,url in SOURCES.items():
        folder=out/name;folder.mkdir(exist_ok=True);pdf=folder/'source.pdf'
        try:
            if not pdf.exists():
                with requests.get(url,stream=True,timeout=(20,60)) as r:
                    r.raise_for_status();temp=folder/'source.pdf.part';size=0
                    with temp.open('wb') as f:
                        for b in r.iter_content(1024*1024):
                            size+=len(b)
                            if size>60_000_000:raise ValueError('公开PDF超出本批预留大小')
                            f.write(b)
                with temp.open('rb') as f:
                    if f.read(5)!=b'%PDF-':raise ValueError('来源未返回PDF，不能当规范解析')
                os.replace(temp,pdf)
            doc=fitz.open(pdf)
            dump(folder/'source.json',{'url':url,'sha256':sha(pdf),'pages':len(doc),'version_claim':name,'status':'原件待目录与封面核对','use':'规范知识来源，非比赛外部图像数据'})
            for index,page in enumerate(doc):
                target=folder/f'page_{index+1:04}.json'
                if target.exists():continue
                embedded=page.get_text().strip()
                if len(embedded)>100:
                    item={'page':index+1,'method':'embedded_text','text':embedded,'blocks':[]}
                else:
                    if engine is None:
                        engine=RapidOCR(det_model_path='/model/RapidOCR/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_server.onnx',
                            rec_model_path='/model/RapidOCR/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_server.onnx',
                            cls_model_path='/model/RapidOCR/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx',
                            intra_op_num_threads=4,inter_op_num_threads=1)
                    pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
                    image=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
                    recognized,_=engine(image)
                    blocks=[{'box':np.asarray(b).tolist(),'text':str(t),'confidence':float(c)} for b,t,c in (recognized or [])]
                    item={'page':index+1,'method':'ocr','text':'\n'.join(b['text'] for b in blocks),'blocks':blocks,'render_size':[pix.width,pix.height]}
                item['thresholds_verified']=False;dump(target,item)
                dump(ROOT/'standards_status.json',{'status':'reading_public_standard','standard':name,'page':index+1,'total_pages':len(doc),'time':time.time()})
                if (index+1)%10==0:print('已读取规范页',name,index+1,'/',len(doc),flush=True)
            results[name]={'status':'pages_extracted_rules_unverified','pages':len(doc)}
        except Exception as e:
            results[name]={'status':'failed','error_type':type(e).__name__}
            print('规范读取失败',name,type(e).__name__,flush=True)
        dump(out/'read_results.json',results)
    dump(ROOT/'standards_status.json',{'status':'source_reading_finished_rule_extraction_pending','results':results,'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'standards_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
