param(
    [ValidateRange(0.05, 0.95)]
    [double]$Confidence = 0.5,
    [string]$Model = 'yolo11l.pt',
    [ValidateSet(640, 768, 960, 1280)]
    [int]$ImageSize = 768,
    [ValidateRange(2, 8)]
    [double]$VramGB = 8
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'The Python environment is missing. See README.md for setup.'
}
# The camera can be opened by only one viewer at a time.
Get-Process ffplay -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -eq 'Thermal Camera - Live' } |
    Stop-Process
& $python (Join-Path $PSScriptRoot 'thermal_people.py') --confidence $Confidence `
    --model (Join-Path $PSScriptRoot ('models\' + $Model)) --device cuda:0 `
    --imgsz $ImageSize --vram-gb $VramGB
exit $LASTEXITCODE
