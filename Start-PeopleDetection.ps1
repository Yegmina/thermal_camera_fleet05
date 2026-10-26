param(
    [ValidateRange(0.05, 0.95)]
    [double]$Confidence = 0.5,
    [string]$Model = 'thermal-yolov5x-adam.pt',
    [ValidateSet(640, 768, 960, 1280)]
    [int]$ImageSize = 640,
    [ValidateRange(128, 1280)]
    [int]$TileSize = 384,
    [ValidateRange(0, 1279)]
    [int]$TileOverlap = 128,
    [ValidateRange(1, 16)]
    [int]$BatchSize = 5,
    [switch]$NoTiles,
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
$arguments = @((Join-Path $PSScriptRoot 'thermal_people.py'), '--confidence', $Confidence,
    '--model', (Join-Path $PSScriptRoot ('models\' + $Model)), '--device', 'cuda:0',
    '--imgsz', $ImageSize, '--vram-gb', $VramGB, '--tile-size', $TileSize,
    '--tile-overlap', $TileOverlap, '--batch-size', $BatchSize)
if ($NoTiles) { $arguments += '--no-tiles' }
& $python @arguments
exit $LASTEXITCODE
