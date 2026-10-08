function Write-LearningHeartbeat {
 param([string]$Path,[string]$Json,[int]$Attempts=50,[int]$PauseMs=100)
 $ErrorActionPreference='Stop'
 $taskDestination=[IO.Path]::GetFullPath($Path)
 $taskTemporary=$taskDestination+'.'+$PID+'.heartbeat.tmp'
 try {
  [IO.File]::WriteAllText($taskTemporary,$Json,(New-Object Text.UTF8Encoding($false)))
  for($taskAttempt=0;$taskAttempt -lt $Attempts;$taskAttempt++) {
   try {
    if([IO.File]::Exists($taskDestination)){[IO.File]::Replace($taskTemporary,$taskDestination,[NullString]::Value)}
    else{[IO.File]::Move($taskTemporary,$taskDestination)}
    return $true
   } catch {
    if($_.Exception.GetBaseException() -isnot [IO.IOException]){throw}
    if($taskAttempt -lt $Attempts-1){Start-Sleep -Milliseconds $PauseMs}
   }
  }
  return $false
 } finally {
  if([IO.File]::Exists($taskTemporary)){Remove-Item -LiteralPath $taskTemporary -Force -ErrorAction SilentlyContinue}
 }
}
