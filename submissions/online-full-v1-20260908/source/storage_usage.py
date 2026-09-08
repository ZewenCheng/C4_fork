"""读取真实目录占用；允许本作业小型候选临时文件原子改名造成的ENOENT。"""
import subprocess,re
def workspace_used():
    for attempt in range(2):
        r=subprocess.run(['du','-s','-B1','/workspace'],capture_output=True,text=True)
        errors=[s for s in r.stderr.splitlines() if s.strip()]
        benign=bool(errors) and all(re.fullmatch(r"du: cannot access '/workspace/work/c4-experiments/expert-autotrain-20260906/pseudo_regions/[0-9a-f]{64}\.(?:npz|json)\.tmp': No such file or directory",s) for s in errors)
        if (r.returncode==0 or benign) and r.stdout.strip():
            # 临时文件改名的少量遗漏另加64MiB余量；不忽略权限或其他目录读取错误。
            return int(r.stdout.split()[0])+(64*1024**2 if benign else 0)
    raise RuntimeError('工作区占用读取异常，不冒充空间足够')
