param([string]$ProjectRoot = $PSScriptRoot, [string]$ResumeTrainerSid = '', [string]$ResumeEvaluatorSid = '')
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator PowerShell is required. No accounts or tasks were changed.'
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$root = Join-Path $ProjectRoot '.runtime\learning_roles'
$admin = Join-Path $root 'admin'
$evidence = Join-Path $root 'evaluator'
$trainerOut = Join-Path $root 'trainer'
$intake = Join-Path $root 'intake'
$source = Join-Path $root 'source'
$release = Join-Path $ProjectRoot '.runtime\validated_ranker_rollout'
trap {
    if ($root -and (Test-Path -LiteralPath $root)) {
        ($_.Exception.Message+"`n"+$_.ScriptStackTrace) | Set-Content -LiteralPath (Join-Path $root 'installation_failure.txt') -Encoding UTF8
    }
    throw
}

function Set-PrivateDirectory([string]$Path, [hashtable]$Grants) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $all = @{'S-1-5-18'='FullControl'; 'S-1-5-32-544'='FullControl'}
    foreach ($key in $Grants.Keys) { $all[$key] = $Grants[$key] }
    foreach ($key in $all.Keys) {
        $sid = New-Object Security.Principal.SecurityIdentifier($key)
        $rule = New-Object Security.AccessControl.FileSystemAccessRule($sid, $all[$key], 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}
foreach ($name in @('GptBotTrainer', 'GptBotEvaluator')) {
    $user = Get-LocalUser -Name $name -ErrorAction SilentlyContinue
    $expectedSid = if ($name -eq 'GptBotTrainer') {$ResumeTrainerSid} else {$ResumeEvaluatorSid}
    if ($user -and $user.SID.Value -ne $expectedSid) { throw "Account $name already exists; refusing to reuse an unverified principal." }
}
Set-PrivateDirectory $root @{$identity.User.Value='FullControl'}
Set-PrivateDirectory $admin @{}
# SecureString credentials are generated in-process; never written as plaintext.
foreach ($name in @('GptBotTrainer', 'GptBotEvaluator')) {
    if (Get-LocalUser -Name $name -ErrorAction SilentlyContinue) { continue }
    $bytes = New-Object byte[] 48
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $password = ConvertTo-SecureString ([Convert]::ToBase64String($bytes)+'aA1!') -AsPlainText -Force
    New-LocalUser -Name $name -Password $password -AccountNeverExpires -PasswordNeverExpires -Description 'Isolated candidate-learning service' | Out-Null
    $password | ConvertFrom-SecureString | Set-Content -LiteralPath (Join-Path $admin "$name.dpapi")
}
$trainerSid = (Get-LocalUser GptBotTrainer).SID.Value
$evaluatorSid = (Get-LocalUser GptBotEvaluator).SID.Value
Set-PrivateDirectory $root @{$identity.User.Value='FullControl'; $trainerSid='ReadAndExecute'; $evaluatorSid='ReadAndExecute'}
Set-PrivateDirectory $admin @{$trainerSid='ReadAndExecute'; $evaluatorSid='ReadAndExecute'}
# Credentials stay in a separate administrator-only child (deployment has no secrets).
$credentials = Join-Path $admin 'credentials'
Set-PrivateDirectory $credentials @{}
foreach ($name in @('GptBotTrainer','GptBotEvaluator')) {
    if (Test-Path -LiteralPath (Join-Path $admin "$name.dpapi")) {
        Move-Item -LiteralPath (Join-Path $admin "$name.dpapi") -Destination $credentials
    }
    $acl = Get-Acl -LiteralPath (Join-Path $credentials "$name.dpapi")
    $acl.SetAccessRuleProtection($false, $false)
    foreach ($rule in @($acl.Access)) { if (-not $rule.IsInherited) { $acl.RemoveAccessRuleSpecific($rule) } }
    Set-Acl -LiteralPath (Join-Path $credentials "$name.dpapi") -AclObject $acl
}
Set-PrivateDirectory $source @{$trainerSid='ReadAndExecute'; $evaluatorSid='ReadAndExecute'}
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'files') -Filter '*.py' -File | Copy-Item -Destination $source
Set-PrivateDirectory $trainerOut @{$trainerSid='Modify'; $evaluatorSid='ReadAndExecute'}
Set-PrivateDirectory $evidence @{$evaluatorSid='Modify'; $identity.User.Value='ReadAndExecute'}
Set-PrivateDirectory $intake @{$trainerSid='ReadAndExecute'; $evaluatorSid='Modify'}
Set-PrivateDirectory $release @{$evaluatorSid='Modify'; $identity.User.Value='ReadAndExecute'}

# Explicit deny on the original raw dataset directory; copied Python source is public.
$files = Join-Path $ProjectRoot 'files'
$acl = Get-Acl -LiteralPath $files
$deny = New-Object Security.AccessControl.FileSystemAccessRule((New-Object Security.Principal.SecurityIdentifier($trainerSid)), 'ReadAndExecute', 'ContainerInherit,ObjectInherit', 'None', 'Deny')
$acl.AddAccessRule($deny)
Set-Acl -LiteralPath $files -AclObject $acl
foreach ($relative in @('.runtime\independent_evaluation')) {
    $path = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $acl = Get-Acl -LiteralPath $path
    $acl.AddAccessRule($deny)
    Set-Acl -LiteralPath $path -AclObject $acl
}
$deployment = @{
    project_root=$ProjectRoot
    trainer_sid=$trainerSid; evaluator_sid=$evaluatorSid
    registry=$evidence; dataset=(Join-Path $files 'critic_dataset_v2.jsonl')
    training_input=(Join-Path $intake 'training.jsonl')
    candidate_output=(Join-Path $trainerOut 'candidate.json'); candidate_input=(Join-Path $trainerOut 'candidate.json')
    trainer_status=(Join-Path $trainerOut 'status.json'); status=(Join-Path $evidence 'status.json')
    bootstrap_cutoff=[DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    portfolio_request=(Join-Path $evidence 'portfolio_request.json'); release_root=$release
}
$deploymentPath = Join-Path $admin 'deployment.json'
$deployment | ConvertTo-Json | Set-Content -LiteralPath $deploymentPath -Encoding UTF8
# Python accepts utf-8-sig for Windows PowerShell's JSON BOM.
$python = Join-Path $ProjectRoot 'pyembed\python.exe'
foreach ($role in @('evaluator','trainer')) {
    $account = if ($role -eq 'trainer') {'GptBotTrainer'} else {'GptBotEvaluator'}
    $minutes = if ($role -eq 'trainer') {60} else {1}
    $action = New-ScheduledTaskAction -Execute $python -Argument "`"$(Join-Path $source 'forward_evidence_service.py')`" --deployment `"$deploymentPath`" --role $role" -WorkingDirectory $(if ($role -eq 'trainer') {$trainerOut} else {$evidence})
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $minutes)
    $taskPrincipal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$account" -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $(if ($role -eq 'trainer') {50} else {5})) -StartWhenAvailable
    Register-ScheduledTask -TaskName "GptBot-$role" -Action $action -Trigger $trigger -Principal $taskPrincipal -Settings $settings | Out-Null
    Start-ScheduledTask -TaskName "GptBot-$role"
}
@{installed_at=[DateTime]::UtcNow.ToString('o'); trainer_sid=$trainerSid; evaluator_sid=$evaluatorSid; state='INSTALLED_NOT_APPROVED'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $admin 'installation.json') -Encoding UTF8
@{installed_at=[DateTime]::UtcNow.ToString('o'); state='INSTALLED_NOT_APPROVED'; tasks=@('GptBot-evaluator','GptBot-trainer')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'installation_public.json') -Encoding UTF8
if (Test-Path -LiteralPath (Join-Path $root 'installation_failure.txt')) {
    Remove-Item -LiteralPath (Join-Path $root 'installation_failure.txt')
}
Write-Output 'Service accounts and schedules installed. No model promotion or BUY enablement was performed.'
