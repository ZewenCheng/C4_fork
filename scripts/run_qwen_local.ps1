param(
    [ValidateSet('check', 'smoke', 'test', 'embed')]
    [string]$Action = 'check',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'
$packageDirectory = Split-Path -Parent $PSScriptRoot
$projectDirectory = Split-Path -Parent $packageDirectory
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw '缺少项目 .venv；请先按本地部署说明安装独立环境。'
}
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    if ($Action -eq 'test') {
        & $pythonExecutable -B -m unittest discover -s (Join-Path $packageDirectory 'tests') -p 'test_qwen_local.py' -v
    } else {
        & $pythonExecutable -B (Join-Path $packageDirectory 'src\qwen_local_cli.py') $Action @RemainingArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "本地 Qwen $Action 执行未通过，退出码 $LASTEXITCODE。"
    }
} finally {
    $env:PYTHONUTF8 = $previousUtf8
}
