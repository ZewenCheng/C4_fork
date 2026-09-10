"""正式伙伴入口参数适配，转交统一整链计时与恢复执行器。"""
from pathlib import Path
import argparse,os,sys,uuid
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='/dataset/决赛数据集1/赛题4/线上测试集')
    p.add_argument('--work',type=Path)
    p.add_argument('--output',type=Path,default=Path('/workspace/result/result'))
    p.add_argument('--resume',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.resume and a.work is None:p.error('恢复必须指定work')
    work=a.work or Path('/workspace/work')/('c4-cv-partner-run-'+uuid.uuid4().hex)
    code=Path(__file__).resolve().parent
    args=[sys.executable,'-B',str(code/'full_partner_entry.py'),'--baseline-code',str(code),'--legacy-root','/workspace/work/road-infrastructure-finals-cloud','--input',a.input,'--work',str(work),'--output',str(a.output)]
    if a.resume:args.append('--resume')
    if a.check:args.append('--check')
    os.execv(sys.executable,args)
if __name__=='__main__':main()
