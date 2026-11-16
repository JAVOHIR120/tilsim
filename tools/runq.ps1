# Run heavy jobs one after another at BelowNormal priority (the owner's rule: one heavy job at a time).
#   powershell -File tools/runq.ps1 -Queue out/q-en.txt
# Each non-empty line of the queue file is "<python script> <args...>", run with the voiceclone venv from E:\voicelab.
# Output of every job goes to out/runq.log (appended), with start/end times and exit codes.
param([Parameter(Mandatory = $true)][string]$Queue)
$py = "E:\tools\voiceclone\.venv\Scripts\python.exe"
$log = "E:\voicelab\out\runq.log"
Set-Location E:\voicelab
$env:PYTHONIOENCODING = "utf-8"
foreach ($line in Get-Content $Queue -Encoding utf8) {
    if (-not $line.Trim() -or $line.TrimStart().StartsWith("#")) { continue }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "=== $stamp START $line" | Out-File $log -Append -Encoding utf8
    $tmp = [System.IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $py -ArgumentList $line -NoNewWindow -PassThru -RedirectStandardOutput "$tmp.out" -RedirectStandardError "$tmp.err"
    try { $p.PriorityClass = "BelowNormal" } catch {}
    $handle = $p.Handle   # keeps the exit code readable
    $p.WaitForExit()
    Get-Content "$tmp.out" -Encoding utf8 | Out-File $log -Append -Encoding utf8
    Get-Content "$tmp.err" -Encoding utf8 | Select-String -Pattern "Error|Traceback|error:" -Context 0,3 | Out-File $log -Append -Encoding utf8
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END exit=$($p.ExitCode) $line" | Out-File $log -Append -Encoding utf8
    Remove-Item "$tmp", "$tmp.out", "$tmp.err" -ErrorAction SilentlyContinue
}
"=== QUEUE DONE $Queue" | Out-File $log -Append -Encoding utf8
