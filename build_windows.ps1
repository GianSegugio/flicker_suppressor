$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = ".\.venv-gui\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        $Version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($Version -ne "3.12") { throw "Python 3.12 is required. Found Python $Version." }
        python -m venv .venv-gui
    } elseif ($py) { py -3.12 -m venv .venv-gui }
    else { throw "Python 3.12 was not found." }
}
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install -r .\requirements.txt -r .\requirements-gui.txt -r .\requirements-build.txt
& $Python .\tools\make_icon.py
if (Test-Path .\build) { Remove-Item -Recurse -Force .\build }
& $Python -m nuitka `
    --standalone `
    --enable-plugin=pyside6 `
    --msvc=latest `
    --windows-console-mode=disable `
    --output-dir=build `
    --output-filename="Flicker Suppressor.exe" `
    --windows-icon-from-ico=.\assets\app.ico `
    --include-package=torch `
    --include-package-data=torch `
    --include-package=numpy `
    --include-package=PIL `
    --include-package=einops `
    --include-module=autosettings `
    --include-data-dir=.\models=models `
    --include-data-dir=.\assets=assets `
    --include-data-file=.\README.md=README.md `
    --include-data-file=.\DOCUMENTATION.md=DOCUMENTATION.md `
    --include-data-file=.\LICENSE=LICENSE `
    --include-data-file=.\RELEASE-LICENSE-THIRD-PARTY=RELEASE-LICENSE-THIRD-PARTY `
    .\gui_main.py
Write-Host "Portable build: $PSScriptRoot\build\gui_main.dist"
