param([switch]$Restart,[switch]$AttachOnly)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython=Join-Path $taskRoot 'pyembed\python.exe'
$taskWorker=Join-Path $taskRoot 'files\rl_headless_worker.py'
$taskLoop=Join-Path $taskRoot 'headless_loop.ps1'
$taskRuntime=Join-Path $taskRoot '.runtime'
$taskStateFile=Join-Path $taskRuntime 'rl_worker_bg.json'
$taskState=Get-Content -LiteralPath $taskStateFile -Raw | ConvertFrom-Json
$taskProcesses=@()
# A stale wrapper receipt must not hide an orphan from a failed prior launch.
$taskOrphans=@(Get-CimInstance Win32_Process | Where-Object {
 ($_.ExecutablePath -eq $taskPython -and $_.CommandLine -like "*$taskWorker*") -or
 ($_.Name -eq 'powershell.exe' -and $_.CommandLine -like "*$taskLoop*")
})
$taskProcessIds=@($taskState.wrapper_pid,$taskState.python_pid)+@($taskOrphans.ProcessId)
foreach($taskProcessId in @($taskProcessIds | Select-Object -Unique)) {
 if(-not $taskProcessId){continue}
 $taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId"
 if($null -eq $taskProcess){continue}
 $taskVerified=(($taskProcess.ExecutablePath -eq $taskPython -and $taskProcess.CommandLine -like "*$taskWorker*") -or
  ($taskProcess.Name -eq 'powershell.exe' -and $taskProcess.CommandLine -like "*$taskLoop*"))
 if(-not $taskVerified){throw "Learning worker PID identity mismatch: $taskProcessId"}
 $taskProcesses+=$taskProcess
}
$taskSummary=[ordered]@{operation='verified_learning_collector_restart';worker_ids=@($taskProcesses.ProcessId);trading_processes_touched=$false}
if($Restart -and $AttachOnly){throw 'Choose Restart or AttachOnly'}
if(-not $Restart -and -not $AttachOnly){$taskSummary | ConvertTo-Json;return}
$taskAttachWorker=0
if($AttachOnly) {
 $taskWrappers=@($taskProcesses | Where-Object {$_.Name -eq 'powershell.exe'})
 $taskWorkers=@($taskProcesses | Where-Object {$_.ExecutablePath -eq $taskPython})
 if($taskWrappers.Count -ne 0 -or $taskWorkers.Count -ne 1){throw 'AttachOnly requires exactly one verified orphan and no active wrapper'}
 $taskAttachWorker=$taskWorkers[0].ProcessId
}
$taskBackup=Join-Path $taskRuntime ('collector_recovery_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
New-Item -ItemType Directory -Path $taskBackup | Out-Null
foreach($taskName in @('rl_worker_bg.json','rl_worker_status.json','collector_integrity.stop','rl_worker_train.lock')) {
 $taskPath=Join-Path $taskRuntime $taskName
 if(Test-Path -LiteralPath $taskPath){Copy-Item -LiteralPath $taskPath -Destination $taskBackup}
}
# Stop verified wrapper first, preventing it from respawning the old worker.
foreach($taskProcess in @($taskProcesses | Where-Object {-not $AttachOnly} | Sort-Object @{Expression={if($_.Name -eq 'powershell.exe'){0}else{1}}})) {
 & taskkill.exe /PID $taskProcess.ProcessId /F | Out-Null
 if($LASTEXITCODE -ne 0 -and (Get-Process -Id $taskProcess.ProcessId -ErrorAction SilentlyContinue)){throw 'Verified learning worker stop failed'}
}
$taskTrainLock=Join-Path $taskRuntime 'rl_worker_train.lock'
if(-not $AttachOnly -and (Test-Path -LiteralPath $taskTrainLock)) {
 $taskLock=Get-Content -LiteralPath $taskTrainLock -Raw | ConvertFrom-Json
 if(-not (Get-Process -Id ([int]$taskLock.pid) -ErrorAction SilentlyContinue)){Remove-Item -LiteralPath $taskTrainLock}
}
# Dataset byte-lock remains untouched. Incident marker is cleared only by a real successful cycle.
$taskArguments=@('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',$taskLoop,'--enable-collector')
if($AttachOnly){$taskArguments=@('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',$taskLoop,'-ExistingPythonPid',$taskAttachWorker,'--enable-collector');$taskSummary.operation='attach_verified_learning_supervisor'}
$taskNew=Start-Process -FilePath 'powershell.exe' -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskBackup 'wrapper.stdout.log') -RedirectStandardError (Join-Path $taskBackup 'wrapper.stderr.log')
$taskSummary.new_wrapper_pid=$taskNew.Id;$taskSummary.incident_backup=$taskBackup
$taskSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBackup 'restart.json') -Encoding UTF8
$taskReady=$false
for($taskTry=0;$taskTry -lt 12;$taskTry++) {
 Start-Sleep -Seconds 1
 $taskNew.Refresh()
 if($taskNew.HasExited){throw "Learning wrapper exited; inspect $taskBackup\wrapper.stderr.log"}
 try {
  $taskLive=Get-Content -LiteralPath $taskStateFile -Raw | ConvertFrom-Json
  if($taskLive.wrapper_pid -eq $taskNew.Id -and $taskLive.state -eq 'running') {
   $taskActual=Get-CimInstance Win32_Process -Filter "ProcessId=$($taskLive.python_pid)"
   if($taskActual.ExecutablePath -eq $taskPython -and $taskActual.CommandLine -like "*$taskWorker*"){$taskReady=$true;$taskSummary.new_python_pid=$taskActual.ProcessId;break}
  }
 } catch { }
}
if(-not $taskReady){throw "Learning wrapper heartbeat unverified; inspect $taskBackup"}
$taskSummary.launch_verified=$true
$taskSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBackup 'restart.json') -Encoding UTF8
$taskSummary | ConvertTo-Json
