$ErrorActionPreference = 'Stop'
$player = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '.tools') -Filter ffplay.exe -Recurse |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $player) {
    throw 'The FFplay viewer is missing from the .tools folder.'
}

& $player -hide_banner -loglevel warning -f dshow -rtbufsize 64M `
    -video_size 640x512 -framerate 50 -i 'video=ThermalCamera' `
    -vf 'normalize=smoothing=30' -window_title 'Thermal Camera - Live' `
    -x 960 -y 768
