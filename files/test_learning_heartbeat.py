import subprocess,tempfile,unittest
from pathlib import Path

class HeartbeatTests(unittest.TestCase):
    def test_atomic_publication_preserves_locked_prior_snapshot_and_recovers(self):
        root=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.ps1';target=Path(d)/'status.json'
            script="""
. 'HELPER'
$taskPath='TARGET'
if(-not (Write-LearningHeartbeat $taskPath '{"generation":1}')){throw 'first failed'}
$taskReader=[IO.File]::Open($taskPath,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
try {
 if(Write-LearningHeartbeat $taskPath '{"generation":2}' -Attempts 3 -PauseMs 1){throw 'locked replace claimed success'}
} finally {$taskReader.Dispose()}
if((Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json).generation -ne 1){throw 'prior snapshot damaged'}
if(-not (Write-LearningHeartbeat $taskPath '{"generation":3}')){throw 'recovery failed'}
if((Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json).generation -ne 3){throw 'fresh snapshot missing'}
if(Get-ChildItem -LiteralPath 'DIRECTORY' -Filter '*.heartbeat.tmp'){throw 'temporary leaked'}
""".replace('HELPER',str(root/'learning_heartbeat.ps1')).replace('TARGET',str(target)).replace('DIRECTORY',d)
            p.write_text(script,encoding='utf-8')
            r=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(p)],capture_output=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stderr.decode(errors='replace'))

if __name__=='__main__':unittest.main()
