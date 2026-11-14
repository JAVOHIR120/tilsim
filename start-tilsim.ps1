# Tilsim: ovoz serveri + internet tunneli.
# Ishga tushirish: start-tilsim.bat ni ikki marta bosing, yoki: powershell -ExecutionPolicy Bypass -File start-tilsim.ps1
# To'xtatish: shu oynada Ctrl+C. Sayt (javohir120.github.io/tilsim) bu oyna ochiq turgan paytda ovoz yaratadi.
#
# 1. engine/serve.py modelni GPU ga yuklaydi (http://127.0.0.1:8790, faqat shu kompyuter ichida).
# 2. cloudflared tekin "quick tunnel" ochadi: https://<tasodifiy>.trycloudflare.com -> 127.0.0.1:8790.
# 3. Tunnel manzili har safar yangi bo'ladi, shuning uchun u gist ga yoziladi; sayt manzilni o'sha yerdan o'qiydi.
# Noutbuk zaryadlovchiga ulangan bo'lsin: batareyada GPU 2 baravar sekin ishlaydi.

param([int]$Port = 8790)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Py = 'E:\tools\voiceclone\.venv\Scripts\python.exe'
$Cf = 'E:\tools\cloudflared\cloudflared.exe'
$Gist = '5801b83645819ab8d22340b5b60798f2'
$Out = Join-Path $Root 'out'
New-Item -ItemType Directory -Force $Out | Out-Null

function Say([string]$text) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $text) }

function Set-ApiAddress($url) {
    # The site and clients read this gist: they try the laptop first, then the cloud copy (deploy/modal_app.py).
    # Only the "laptop" field changes here; the cloud address stays.
    $value = if ($url) { $url } else { 'none' }
    python (Join-Path $Root 'tools\set_api_address.py') --laptop $value | Out-Null
}

function Stop-Tree($proc) {
    if ($proc -and -not $proc.HasExited) { taskkill /PID $proc.Id /T /F 2>$null | Out-Null }
}

function Start-Tunnel {
    $log = Join-Path $Out 'tunnel.log'
    if (Test-Path $log) { Remove-Item $log -Force }
    $p = Start-Process -FilePath $Cf -ArgumentList @('tunnel', '--no-autoupdate', '--metrics', '127.0.0.1:20241',
            '--url', "http://127.0.0.1:$Port") `
        -RedirectStandardError $log -RedirectStandardOutput (Join-Path $Out 'tunnel.out.log') -WindowStyle Hidden -PassThru
    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $log) {
            $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { Stop-Tree $p; throw "tunnel did not start; see $log" }
    # Wait for cloudflared's own /ready (a live connection to Cloudflare). Do NOT look the new name up here:
    # asking DNS before the record has spread makes the local resolver cache "no such name" for minutes
    # (seen 2026-11-15 on a phone hotspot), and then this laptop's own browser cannot reach the site's voice.
    $ok = $false
    for ($i = 0; $i -lt 40 -and -not $ok; $i++) {
        try { $ok = (Invoke-WebRequest 'http://127.0.0.1:20241/ready' -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ok) { Stop-Tree $p; throw "tunnel did not connect; see $log" }
    return @{ Proc = $p; Url = $url }
}

$server = $null; $tunnel = $null
try {
    if (-not (Test-Path $Cf)) { throw "cloudflared not found at $Cf" }
    $battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
    if ($battery -and $battery.BatteryStatus -ne 2) { Say 'DIQQAT: noutbuk batareyada. Zaryadlovchini ulang, aks holda ovoz juda sekin chiqadi.' }

    Say 'Ovoz serveri ishga tushmoqda (model GPU ga yuklanadi, 1-2 daqiqa)...'
    $server = Start-Process -FilePath $Py -ArgumentList @('-X', 'utf8', 'engine\serve.py', '--port', $Port) `
        -WorkingDirectory $Root -WindowStyle Minimized -PassThru
    $ready = $false
    for ($i = 0; $i -lt 180 -and -not $ready; $i++) {
        Start-Sleep -Seconds 2
        if ($server.HasExited) { throw "server stopped; see out\serve.log" }
        try { $ready = (Invoke-RestMethod "http://127.0.0.1:$Port/v1/health" -TimeoutSec 3).ready } catch { }
    }
    if (-not $ready) { throw 'model did not load in 6 minutes; see out\serve.log' }
    Say 'Model tayyor.'

    $tunnel = Start-Tunnel
    Set-ApiAddress $tunnel.Url
    Say "Tunnel ochiq: $($tunnel.Url)"
    Say 'Sayt endi ovoz yaratadi: https://javohir120.github.io/tilsim/   (to''xtatish: Ctrl+C)'

    while ($true) {
        Start-Sleep -Seconds 10
        if ($server.HasExited) { throw 'server stopped; see out\serve.log' }
        if ($tunnel.Proc.HasExited) {
            Say 'Tunnel uzildi, qayta ochilmoqda...'
            $tunnel = Start-Tunnel
            Set-ApiAddress $tunnel.Url
            Say "Yangi tunnel: $($tunnel.Url)"
        }
    }
}
finally {
    Say 'To''xtatilmoqda...'
    try { Set-ApiAddress $null } catch { }
    if ($tunnel) { Stop-Tree $tunnel.Proc }
    Stop-Tree $server
    Say 'Server va tunnel yopildi. Sayt endi "server o''chiq" deb ko''rsatadi.'
}
