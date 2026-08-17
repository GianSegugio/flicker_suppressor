$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = ".\.venv-gui\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python 3.12 was not found on PATH." }
    $Version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($Version -ne "3.12") { throw "Python 3.12 is required. Found $Version." }
    python -m venv .venv-gui
}
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install -r .\requirements.txt -r .\requirements-gui.txt
& $Python .\gui_main.py
