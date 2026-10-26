$ErrorActionPreference = 'Stop'
$uv = Join-Path $PSScriptRoot '.tools\uv\uv.exe'
$environment = Join-Path $PSScriptRoot '.venv-gpu'
$python = Join-Path $environment 'Scripts\python.exe'
function Get-ThermalDownload {
    param([string]$Url, [string]$Destination)
    & curl.exe --fail --location --retry 3 --silent --show-error --output $Destination $Url
    if ($LASTEXITCODE) { throw "Download failed: $Url" }
}
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '.tools'), (Join-Path $PSScriptRoot 'models') | Out-Null
if (-not (Test-Path -LiteralPath $uv)) {
    $uvArchive = Join-Path $PSScriptRoot '.tools\uv.zip'
    Get-ThermalDownload 'https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip' $uvArchive
    Expand-Archive -LiteralPath $uvArchive -DestinationPath (Split-Path $uv) -Force
}
$ffmpeg = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '.tools') -Filter ffmpeg.exe -Recurse | Select-Object -First 1
if (-not $ffmpeg) {
    $cameraArchive = Join-Path $PSScriptRoot '.tools\ffmpeg-release-essentials.zip'
    Get-ThermalDownload 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' $cameraArchive
    & tar.exe -xf $cameraArchive -C (Join-Path $PSScriptRoot '.tools')
    if ($LASTEXITCODE) { throw 'Could not unpack the camera tools.' }
}
$models = @(
    @{
        File = 'thermal-detr-r50.bin'
        Url = 'https://huggingface.co/faldeus0092/detr-finetuned-thermal-dogs-and-people/resolve/f7c9ff88dfb256c9fdb36e292046439c7f97e21c/pytorch_model.bin?download=true'
        Hash = '8c10905ad0d26d046eddb664da6cec504fb2ba2c2e7bd8d2306bc926a99cf909'
    },
    @{
        File = 'thermal-yolo11n.pt'
        Url = 'https://huggingface.co/banghyunmin/thermal-people-yolov11n/resolve/fbeb08a979248fe3cbb2756f95401da98103956d/best.pt'
        Hash = 'fdad65ed79d7fd4371e038aa3e43c7ac168addb40f256e316d9f00a78dfbf9a2'
    },
    @{
        File = 'thermal-person.pt'
        Url = 'https://huggingface.co/pitangent-ds/YOLOv8-human-detection-thermal/resolve/a5d30f1e8e185a54a83cb6288d38c806e303bdd2/model.pt'
        Hash = '63181e6c024a8a218020932b516bfb9547130e03e07c8d4671cb2973df038bef'
    }
)
$legacyRuntime = Join-Path $PSScriptRoot '.tools\yolov5-7.0\models\yolo.py'
if (-not (Test-Path -LiteralPath $legacyRuntime)) {
    $legacyArchive = Join-Path $PSScriptRoot '.tools\yolov5-v7.zip'
    Get-ThermalDownload 'https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip' $legacyArchive
    if ((Get-FileHash -LiteralPath $legacyArchive).Hash -ne 'a40568a7979c4b27195d198987e49610de86b038b044f2521f5515455571f852') {
        throw 'The YOLOv5 runtime archive failed its checksum check.'
    }
    Expand-Archive -LiteralPath $legacyArchive -DestinationPath (Join-Path $PSScriptRoot '.tools') -Force
}
foreach ($model in $models) {
    $modelPath = Join-Path $PSScriptRoot ('models\' + $model.File)
    if ((Test-Path -LiteralPath $modelPath) -and (Get-FileHash -LiteralPath $modelPath).Hash -eq $model.Hash) {
        continue
    }
    $temporaryModel = $modelPath + '.download'
    Get-ThermalDownload $model.Url $temporaryModel
    if ((Get-FileHash -LiteralPath $temporaryModel).Hash -ne $model.Hash) {
        throw "Model checksum mismatch: $($model.File)"
    }
    Move-Item -LiteralPath $temporaryModel -Destination $modelPath -Force
}
if (-not (Test-Path -LiteralPath $python)) {
    & $uv venv --python 3.12 $environment
    if ($LASTEXITCODE) { throw 'Could not create the GPU environment.' }
}
& $uv --system-certs pip install --python $python torch==2.9.1 torchvision==0.24.1 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'Could not install CUDA PyTorch.' }
& $uv --system-certs pip install --python $python -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'Could not install the detection dependencies.' }
& $python (Join-Path $PSScriptRoot 'download_thermal_model.py')
if ($LASTEXITCODE) { throw 'Could not download and verify the thermal YOLOv5x checkpoint.' }
& $python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(16).cuda(); print(torch.cuda.get_device_name(0)); print(x.sum().item())'
if ($LASTEXITCODE) { throw 'The CUDA verification failed.' }
