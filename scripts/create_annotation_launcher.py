"""Create a local Windows annotation launcher without starting or reading labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POWERSHELL = r'''param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot 'annotation_launcher_config.json'
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$port = [int]$config.port
if ($port -lt 1 -or $port -gt 65535) { throw 'Invalid local port.' }
$url = "http://127.0.0.1:$port/"

function Read-AnnotationHealth {
    $response = $null
    $reader = $null
    try {
        $request = [System.Net.HttpWebRequest]::Create($url + 'health')
        $request.Proxy = $null
        $request.AllowAutoRedirect = $false
        $request.Timeout = 1500
        $request.ReadWriteTimeout = 1500
        $response = $request.GetResponse()
        if ([int]$response.StatusCode -ne 200) { return $null }
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        $body = $reader.ReadToEnd() | ConvertFrom-Json
        if ($body.kind -eq 'human_annotation' -and $body.ready -eq $true) { return $body }
        return $null
    } catch { return $null }
    finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
    }
}

function Test-LoopbackPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $port)
        if (-not $task.Wait(1000)) { return $false }
        return $client.Connected
    } catch { return $false }
    finally { $client.Dispose() }
}

# Reuse only the declared annotation service. Never stop a process occupying a port.
if ($null -ne (Read-AnnotationHealth)) {
    if ($CheckOnly) { @{status='reused_existing';url=$url;started=$false} | ConvertTo-Json -Compress }
    else { Write-Host "标注服务已经运行，无需重复启动。`n打开：$url`n已保存的进度保持不变。" }
    exit 0
}
if (Test-LoopbackPort) {
    Write-Error "端口 $port 已由其他或未就绪的程序占用。为保护已有程序，本脚本不会终止或替换它。请联系 Codex 检查。"
    exit 3
}

$pythonPath = (Resolve-Path -LiteralPath $config.python -ErrorAction Stop).Path
$repositoryPath = (Resolve-Path -LiteralPath $config.repository -ErrorAction Stop).Path
$dataPath = (Resolve-Path -LiteralPath $config.data -ErrorAction Stop).Path
$scriptPath = Join-Path $repositoryPath 'scripts\serve_review_annotation.py'
foreach ($required in @($scriptPath, (Join-Path $dataPath 'annotation_reviews.jsonl'), (Join-Path $dataPath 'annotation_taxonomy.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "找不到所需文件：$required" }
}
if ($CheckOnly) {
    @{status='ready_to_start';url=$url;started=$false;python=$pythonPath;data=$dataPath} | ConvertTo-Json -Compress
    exit 0
}

# Unique files preserve earlier logs and all annotation data.
$logs = Join-Path $dataPath 'launcher_logs'
[System.IO.Directory]::CreateDirectory($logs) | Out-Null
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$stdout = Join-Path $logs ($runId + '.stdout.log')
$stderr = Join-Path $logs ($runId + '.stderr.log')
$scriptArgument = '"' + $scriptPath + '"'
$dataArgument = '"' + $dataPath + '"'
$serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', $scriptArgument, '--data', $dataArgument, '--port', [string]$port) -WorkingDirectory $repositoryPath -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
@{pid=$serverProcess.Id;port=$port;url=$url;stdout=$stdout;stderr=$stderr;created_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs ($runId + '.launch.json')) -Encoding UTF8
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($null -ne (Read-AnnotationHealth)) {
        Write-Host "标注服务已启动，仅本机可访问。`n打开：$url`n也可双击同目录的 开始标注.html。`n关闭这个提示窗口不会删除或重置标注进度。"
        exit 0
    }
    $serverProcess.Refresh()
    if ($serverProcess.HasExited) { break }
    Start-Sleep -Milliseconds 500
}
Write-Error "服务未在等待时间内就绪。没有终止任何进程。请把日志位置交给 Codex：$logs"
exit 4
'''

CMD = r'''@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_annotation.ps1"
if errorlevel 1 (
  echo.
  echo 启动未完成。请保留上面的提示，联系 Codex 检查；不要删除数据或终止其他程序。
  pause
  exit /b 1
)
echo.
echo 请打开同目录的“开始标注.html”，进入标注页面。
pause
'''

GUIDE = """Chattermill 本机标注启动说明

1. 平时双击“开始标注.html”，点击“打开标注页面”。
2. 如果网页打不开，双击同目录的“启动标注.cmd”。无需管理员权限。
3. 出现“已经运行”或“已启动”后，再打开“开始标注.html”。
4. 标注进度保存在本目录 human_labels 子目录。启动脚本不清空、不覆盖标签；关闭网页不影响已保存的进度。

服务只监听本机 127.0.0.1:8769，不发布到互联网，也不会设置开机自动启动。
若 8769 被其他程序占用，脚本拒绝启动，不终止任何已有进程。
每次实际启动均使用新的日志文件，保存在 launcher_logs 子目录。出现错误时把提示或日志路径交给 Codex。
不要删除、移动或改写 human_labels、annotation_reviews.jsonl、annotation_taxonomy.json。

该启动入口沿用已经安装的本地 Python 环境，不会自动安装依赖或下载模型。
以后若移动项目目录或更换 Python 环境，需要由 Codex 重新生成启动配置。
"""


def update_entry_guide(data: Path) -> bool:
    """Add restart guidance without touching review inputs or annotation stores."""
    entry = data / "开始标注.html"
    if not entry.is_file():
        return False
    current = entry.read_text(encoding="utf-8")
    marker = 'data-local-launcher="v1"'
    if marker in current:
        return False
    old = "如果打不开，通知 Codex 重启本地标注服务即可，已保存标签不会丢失。"
    if old not in current:
        raise ValueError("Entry guide changed; review it before adding startup guidance")
    replacement = ('<span data-local-launcher="v1">如果打不开，双击同目录的 '
                   '<strong>启动标注.cmd</strong>，看到“已经运行”或“已启动”后再打开标注页面。'
                   '脚本会保留已有标签；无需管理员权限，不设置开机自动启动。'
                   '详细步骤见同目录的“标注启动说明.txt”。</span>')
    backup = data / "开始标注_添加启动入口前.html"
    if not backup.exists():
        backup.write_text(current, encoding="utf-8", newline="")
    entry.write_text(current.replace(old, replacement, 1), encoding="utf-8", newline="")
    return True


def create_launcher(data: Path, python: Path, *, port: int = 8769) -> dict:
    data = data.resolve(strict=True)
    python = python.resolve(strict=True)
    if not python.is_file() or not (1 <= port <= 65535):
        raise ValueError("Valid existing Python executable and port required")
    for name in ("annotation_reviews.jsonl", "annotation_taxonomy.json"):
        if not (data / name).is_file():
            raise ValueError(f"Missing input file: {name}")
    # File existence only: this generator never reads reviews or human gold.
    values = {
        "start_annotation.ps1": (POWERSHELL, "utf-8-sig"),
        "启动标注.cmd": (CMD, "utf-8"),
        "标注启动说明.txt": (GUIDE.replace("8769", str(port)), "utf-8-sig"),
        "annotation_launcher_config.json": (json.dumps({"schema": "local_annotation_launcher_v1",
            "python": str(python), "repository": str(ROOT), "data": str(data), "port": port,
            "host": "127.0.0.1", "existing_service_kind": "human_annotation"}, ensure_ascii=False, indent=2)+"\n", "utf-8"),
    }
    for name, (text, encoding) in values.items():
        path = data / name
        if path.exists() and path.read_text(encoding=encoding).replace("\r\n", "\n") != text:
            raise ValueError(f"Existing launcher differs; preserve it and review before changing: {name}")
    for name, (text, encoding) in values.items():
        path = data / name
        if not path.exists():
            path.write_text(text, encoding=encoding, newline="\r\n" if name.endswith((".ps1", ".cmd")) else "\n")
    guide_updated = update_entry_guide(data)
    return {"files": list(values), "data": str(data), "service_started": False,
            "labels_read": False, "entry_guide_updated": guide_updated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    print(json.dumps(create_launcher(args.data, args.python, port=args.port), ensure_ascii=True))


if __name__ == "__main__":
    main()
