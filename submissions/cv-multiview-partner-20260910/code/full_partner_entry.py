"""完整基线推理后追加伙伴复核，隔离工作区并实测整入口秒数。"""
import time
begin_time=time.time()
from pathlib import Path
import argparse,os,signal,subprocess,sys,uuid,math,shutil
from report_contract import read_json,atomic_json,digest_file,digest_json,run_lock


def owned_processes(token):
    """按私有运行标记找子进程，包含另建会话或已被重新托管的裁块进程。"""
    marker=('C4_PARTNER_STAGE_TOKEN='+token).encode()
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid==os.getuid() and marker in (proc/'environ').read_bytes().split(b'\0'):
                found.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return found


def stop_stage(child,token):
    # 不依赖父子关系或会话编号；先停止派生，再终止本次标记下的进程。
    for sig in (signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT,signal.SIGKILL):
        for pid in owned_processes(token):
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        if sig==signal.SIGCONT:time.sleep(.2)
    if child.poll() is None:child.kill()
    child.wait(timeout=10)


def run_stage(command,directory,name,timeout):
    started=time.time()
    with (directory/(name+'.log')).open('ab') as log:
        token=uuid.uuid4().hex
        env=dict(os.environ,C4_PARTNER_STAGE_TOKEN=token)
        child=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True,env=env)
        try:code=child.wait(timeout=timeout)
        except BaseException:
            stop_stage(child,token)
            raise
        if owned_processes(token):
            stop_stage(child,token)
            raise RuntimeError(name+'退出后遗留子进程，已清理')
    if code:raise RuntimeError(name+'失败，退出码'+str(code))
    return {'stage':name,'seconds':time.time()-started}


def input_snapshot(directory):
    directory=directory.resolve()
    if not directory.is_dir():raise ValueError('输入目录不存在')
    return {p.relative_to(directory).as_posix():digest_file(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_complete(work,output,config):
    previous=read_json(work/'receipt.json')
    elapsed=previous.get('infer_time')
    if (previous.get('status')!='complete' or previous.get('configuration_sha256')!=digest_json(config)
        or previous.get('output')!=str(output) or type(elapsed) not in (float,int)
        or not math.isfinite(elapsed) or elapsed<=0
        or previous.get('result_sha256')!=digest_file(output/'result.json')
        or previous.get('infer_time_sha256')!=digest_file(output/'infer_time.json')
        or read_json(output/'infer_time.json')!={'infer_time':elapsed}):
        raise ValueError('完整入口完成凭证或成对产物被修改')
    return previous


def main():
    def cancel(_signal,_frame):raise KeyboardInterrupt('完整入口收到取消信号')
    signal.signal(signal.SIGTERM,cancel)
    parser=argparse.ArgumentParser()
    for name in ['baseline-code','legacy-root','input','work','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();work=args.work.resolve();output=args.output.resolve()
    boundary=Path('/workspace/work').resolve()
    official_output=Path('/workspace/result/result').resolve()
    if work==boundary or not work.is_relative_to(boundary) or output==work or (not output.is_relative_to(work) and output!=official_output):
        raise ValueError('完整伙伴输出必须位于独立工作目录内')
    if work.exists() and not args.resume:raise FileExistsError('工作目录已存在，不能覆盖历史版本')
    source=Path(__file__).parent
    config={'baseline_code':str(args.baseline_code.resolve()),'legacy_root':str(args.legacy_root.resolve()),'output':str(output),
        'input':str(args.input.resolve()),'baseline_manifest':digest_file(args.baseline_code/'source_manifest.json'),
        'baseline_models':digest_file(args.baseline_code/'model_manifest.json'),'input_files':input_snapshot(args.input),
        'legacy_sources':{name:digest_file(args.legacy_root/name) for name in
            ['src/common.py','src/extract_features.py','src/qwen_adapter.py','adapters/qwen_local_provider.py',
             'config/deploy_config.json','models/MODEL_MANIFEST.json','models/classifier/expq_lr.npz']},
        'source':{p.name:digest_file(p) for p in source.glob('*.py')},'mode':'cv_multiview_primary_with_closed_partner_revision_gate',
        'baseline_timeout':12000,'partner_timeout':12000}
    if args.resume and (not (work/'configuration.json').is_file() or read_json(work/'configuration.json')!=config):
        raise ValueError('恢复必须使用已有且完全一致的入口配置，不接管不明目录')
    if args.check:
        if not (source/'partner_pipeline.py').is_file():raise ValueError('缺少伙伴流水入口')
        compile((source/'partner_pipeline.py').read_text(encoding='utf-8'),'partner_pipeline.py','exec')
        cmd=[sys.executable,'-B',str(args.baseline_code/'submission_entry.py'),'--check','--input',str(args.input),
             '--work',str(work/'baseline'),'--output',str(work/'baseline_result'),'--report-batch-size','3','--cpu-workers','4']
        if args.resume and (work/'baseline').exists():cmd.append('--resume')
        subprocess.run(cmd,check=True,timeout=180)
        if not (args.legacy_root/'src/extract_features.py').is_file():raise ValueError('缺少旧v2特征入口')
        print('完整伙伴入口静态与基线资产检查通过，尚未执行全量推理');return
    work.mkdir(parents=True,exist_ok=True)
    with run_lock(work/'entry.lock'):
        cfg=work/'configuration.json'
        if cfg.exists() and read_json(cfg)!=config:raise ValueError('完整伙伴恢复版本变化')
        if (work/'receipt.json').exists():
            verify_complete(work,output,config)
            print('完整伙伴结果已核验复用，保留原运行时间，未重新计为完整推理');return
        atomic_json(cfg,config);phases=[]
        baseline_cmd=[sys.executable,'-B',str(args.baseline_code/'submission_entry.py'),'--input',str(args.input),
            '--work',str(work/'baseline'),'--output',str(work/'baseline_result'),'--report-batch-size','3','--cpu-workers','4']
        if args.resume and (work/'baseline').exists():baseline_cmd.append('--resume')
        try:
            phases.append(run_stage(baseline_cmd,work,'baseline',12000))
            online=work/'baseline/candidate_online_20260908'
            phases.append(run_stage([sys.executable,'-B',str(source/'partner_pipeline.py'),'--manifest',str(online/'input_manifest.json'),
                '--packets',str(online/'packets'),'--baseline',str(work/'baseline_result/result.json'),
                '--baseline-receipt',str(work/'baseline/published_result_receipt.json'),
                '--labels',str(work/'baseline/data/vocabulary.json'),'--legacy-root',str(args.legacy_root),
                '--output',str(work/'partner')],work,'partner',12000))
            values=read_json(work/'partner/result.json');base=read_json(work/'baseline_result/result.json')
            if values!=base:raise ValueError('采用门关闭却改变了基线输出')
            if input_snapshot(args.input)!=config['input_files']:raise ValueError('运行期间输入内容变化')
            output.mkdir(parents=True,exist_ok=True)
            if any((output/name).exists() for name in ('result.json','infer_time.json')):
                if output==official_output and not args.resume:
                    archive=work/'previous_official_result';archive.mkdir(exist_ok=False)
                    for name in ('result.json','infer_time.json'):
                        if (output/name).exists():
                            shutil.copy2(output/name,archive/name)
                            if digest_file(output/name)!=digest_file(archive/name):raise ValueError('旧结果保留失败')
                # 中断于发布中途时只接受本次已验证的相同结果，不覆盖不明产物。
                elif not args.resume or not (output/'result.json').exists() or read_json(output/'result.json')!=values:
                    raise FileExistsError('整入口存在不明完成产物，不覆盖')
            atomic_json(output/'result.json',values)
            elapsed=time.time()-begin_time
            atomic_json(output/'infer_time.json',{'infer_time':elapsed})
            atomic_json(work/'receipt.json',{'status':'complete','cases':len(values),'phases':phases,'infer_time':elapsed,
                'result_sha256':digest_file(output/'result.json'),'scope':'本次完整脚本开始至结果准备结束；恢复只计本次调用，不冒充冷启动',
                'infer_time_sha256':digest_file(output/'infer_time.json'),'configuration_sha256':digest_json(config),'output':str(output),
                'accuracy_gain_proven':False,'adoption_enabled':False})
        except BaseException as exc:
            atomic_json(work/'failure.json',{'status':'failed','error_type':type(exc).__name__,'phases':phases,'elapsed_seconds':time.time()-begin_time})
            raise


if __name__=='__main__':main()
