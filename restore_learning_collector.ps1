param([switch]$Restart)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython=Join-Path $taskRoot 'pyembed\python.exe'
$taskWorker=Join-Path $taskRoot 'files\rl_headless_worker.py'
$taskLoop=Join-Path $taskRoot 'headless_loop.ps1'
$taskRuntime=Join-Path $taskRoot '.runtime'
$taskStateFile=Join-Path $taskRuntime 'rl_worker_bg.json'
$taskState=Get-Content -LiteralPath $taskStateFile -Raw | ConvertFrom-Json
$taskProcesses=@()
foreach($taskProcessId in @($taskState.wrapper_pid,$taskState.python_pid)) {
 if(-not $taskProcessId){continue}
 $taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId"
 if($null -eq $taskProcess){continue}
 $taskVerified=(($taskProcess.ExecutablePath -eq $taskPython -and $taskProcess.CommandLine -like "*$taskWorker*") -or
  ($taskProcess.Name -eq 'powershell.exe' -and $taskProcess.CommandLine -like "*$taskLoop*"))
 if(-not $taskVerified){throw "Learning worker PID identity mismatch: $taskProcessId"}
 $taskProcesses+=$taskProcess
}
$taskSummary=[ordered]@{operation='verified_learning_collector_restart';worker_ids=@($taskProcesses.ProcessId);trading_processes_touched=$false}
if(-not $Restart){$taskSummary | ConvertTo-Json;return}
$taskBackup=Join-Path $taskRuntime ('collector_recovery_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
New-Item -ItemType Directory -Path $taskBackup | Out-Null
foreach($taskName in @('rl_worker_bg.json','rl_worker_status.json','collector_integrity.stop','rl_worker_train.lock')) {
 $taskPath=Join-Path $taskRuntime $taskName
 if(Test-Path -LiteralPath $taskPath){Copy-Item -LiteralPath $taskPath -Destination $taskBackup}
}
# Stop verified wrapper first, preventing it from respawning the old worker.
foreach($taskProcess in $taskProcesses) {
 & taskkill.exe /PID $taskProcess.ProcessId /F | Out-Null
 if($LASTEXITCODE -ne 0 -and (Get-Process -Id $taskProcess.ProcessId -ErrorAction SilentlyContinue)){throw 'Verified learning worker stop failed'}
}
$taskTrainLock=Join-Path $taskRuntime 'rl_worker_train.lock'
if(Test-Path -LiteralPath $taskTrainLock) {
 $taskLock=Get-Content -LiteralPath $taskTrainLock -Raw | ConvertFrom-Json
 if(-not (Get-Process -Id ([int]$taskLock.pid) -ErrorAction SilentlyContinue)){Remove-Item -LiteralPath $taskTrainLock}
}
# Dataset byte-lock remains untouched. Incident marker is cleared only by a real successful cycle.
$taskArguments=@('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',$taskLoop,'--enable-collector')
$taskNew=Start-Process -FilePath 'powershell.exe' -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru
$taskSummary.new_wrapper_pid=$taskNew.Id;$taskSummary.incident_backup=$taskBackup
$taskSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBackup 'restart.json') -Encoding UTF8
$taskSummary | ConvertTo-Json
