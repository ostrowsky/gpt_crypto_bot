param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'pyembed\python.exe'
$runner = Join-Path $PSScriptRoot 'files\local_learning_runtime.py'
if ($Stop) { & $python $runner --stop; exit $LASTEXITCODE }
# No RunAs, account provisioning, global tasks or protected ACL modification.
Start-Process -FilePath $python -ArgumentList @('"'+$runner+'"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
Write-Output 'Local logical learning supervisor launched. Read .runtime\learning_roles_local\supervisor.json for verified process status.'
