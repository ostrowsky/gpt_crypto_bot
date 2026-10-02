param([string]$ProjectRoot = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator elevation required; no changes made.'
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$account = 'GptBotCoverageAuthority'
$root = Join-Path $env:ProgramData $account
$evaluator = Get-LocalUser -Name GptBotEvaluator -ErrorAction Stop
$trainer = Get-LocalUser -Name GptBotTrainer -ErrorAction Stop
if ((Get-LocalUser -Name $account -ErrorAction SilentlyContinue) -or (Test-Path -LiteralPath $root)) {
    throw 'Existing authority account/directory: refusing reuse or key replacement.'
}
# Mandatory OS recovery barrier, before every privileged mutation.
$description = 'GptBotCoverageAuthority-' + (Get-Date -Format 'yyyyMMddTHHmmss')
Checkpoint-Computer -Description $description -RestorePointType MODIFY_SETTINGS -ErrorAction Stop
$point = Get-ComputerRestorePoint -ErrorAction Stop | Where-Object { $_.Description -eq $description } |
    Sort-Object SequenceNumber -Descending | Select-Object -First 1
if (-not $point) { throw 'New Windows restore point not verified; installation aborted.' }

function Set-AuthorityAcl([string]$Path, [hashtable]$Grants) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
    $all = @{'S-1-5-18'='FullControl'; 'S-1-5-32-544'='FullControl'}
    foreach ($sid in $Grants.Keys) { $all[$sid] = $Grants[$sid] }
    foreach ($sid in $all.Keys) {
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier($sid)), $all[$sid],
            'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.Principal;
public static class CoverageBatchRights {
    [StructLayout(LayoutKind.Sequential)] struct Attr {
        public uint Length; public IntPtr Root, Name; public uint Flags;
        public IntPtr Descriptor, Quality;
    }
    [StructLayout(LayoutKind.Sequential)] struct Str {
        public ushort Length, MaximumLength; public IntPtr Buffer;
    }
    [DllImport("advapi32.dll")] static extern uint LsaOpenPolicy(IntPtr name, ref Attr a, uint access, out IntPtr p);
    [DllImport("advapi32.dll")] static extern uint LsaAddAccountRights(IntPtr p, IntPtr sid, Str[] r, uint count);
    [DllImport("advapi32.dll")] static extern uint LsaNtStatusToWinError(uint s);
    [DllImport("advapi32.dll")] static extern uint LsaClose(IntPtr p);
    static void Check(uint s) { if (s != 0) throw new Win32Exception((int)LsaNtStatusToWinError(s)); }
    public static void Grant(string id, string right) {
        var sid = new SecurityIdentifier(id); var bytes = new byte[sid.BinaryLength]; sid.GetBinaryForm(bytes, 0);
        IntPtr bin = Marshal.AllocHGlobal(bytes.Length), txt = Marshal.StringToHGlobalUni(right), p = IntPtr.Zero;
        try {
            Marshal.Copy(bytes, 0, bin, bytes.Length);
            var a = new Attr { Length=(uint)Marshal.SizeOf(typeof(Attr)) };
            Check(LsaOpenPolicy(IntPtr.Zero, ref a, 0x810, out p));
            var r = new Str { Length=(ushort)(right.Length*2), MaximumLength=(ushort)((right.Length+1)*2), Buffer=txt };
            Check(LsaAddAccountRights(p, bin, new[] {r}, 1));
        } finally { if (p != IntPtr.Zero) LsaClose(p); Marshal.FreeHGlobal(bin); Marshal.FreeHGlobal(txt); }
    }
}
'@
$created = $false
$directoryCreated = $false
try {
    New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
    $directoryCreated = $true
    Set-AuthorityAcl $root @{}
    $bytes = New-Object byte[] 48
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $password = ConvertTo-SecureString ([Convert]::ToBase64String($bytes)+'aA1!') -AsPlainText -Force
    [Array]::Clear($bytes, 0, $bytes.Length)
    $user = New-LocalUser -Name $account -Password $password -AccountNeverExpires -PasswordNeverExpires -Description 'Independent coverage certificate authority'
    $created = $true
    $sid = $user.SID.Value
    Add-LocalGroupMember -Group (Get-LocalGroup -SID S-1-5-32-545) -Member $account
    foreach ($right in @('SeBatchLogonRight','SeDenyInteractiveLogonRight','SeDenyRemoteInteractiveLogonRight')) {
        [CoverageBatchRights]::Grant($sid, $right)
    }
    Set-AuthorityAcl $root @{$sid='ReadAndExecute'; $evaluator.SID.Value='ReadAndExecute'}
    $private = Join-Path $root 'private'
    $public = Join-Path $root 'public'
    $admin = Join-Path $root 'admin'
    Set-AuthorityAcl $private @{$sid='ReadAndExecute'}
    Set-AuthorityAcl $public @{$sid='ReadAndExecute'; $evaluator.SID.Value='ReadAndExecute'}
    Set-AuthorityAcl $admin @{}
    $password | ConvertFrom-SecureString | Set-Content -LiteralPath (Join-Path $admin 'credential.dpapi') -Encoding ASCII
    $password = $null
    $rsa = [Security.Cryptography.RSA]::Create()
    try {
        $rsa.KeySize = 3072
        $rsa.ToXmlString($true) | Set-Content -LiteralPath (Join-Path $private 'signing_key.xml') -Encoding ASCII
        $rsa.ToXmlString($false) | Set-Content -LiteralPath (Join-Path $public 'verification_key.xml') -Encoding ASCII
    } finally { $rsa.Dispose() }
    $privateAcl = Get-Acl -LiteralPath $private
    $allowed = @('S-1-5-18','S-1-5-32-544',$sid)
    foreach ($rule in $privateAcl.Access) {
        if ($rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -notin $allowed) {
            throw 'Private key ACL contains an unexpected principal.'
        }
    }
    $fingerprint = (Get-FileHash -LiteralPath (Join-Path $public 'verification_key.xml') -Algorithm SHA256).Hash
    $result = @{state='PROVISIONED_NOT_CERTIFYING'; account=$account; authority_sid=$sid;
        restore_sequence=$point.SequenceNumber; restore_description=$description;
        public_key_sha256=$fingerprint; private_key_shared=$false; runtime_eligible=$false;
        signing_task_installed=$false; installed_at=[DateTime]::UtcNow.ToString('o')}
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $public 'installation.json') -Encoding UTF8
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ProjectRoot '.runtime\coverage_authority_installation.json') -Encoding UTF8
    Write-Output 'Coverage authority provisioned with verified restore point. No certificates or model activation.'
} catch {
    if ($created) { Remove-LocalUser -Name $account -ErrorAction SilentlyContinue }
    $resolved = [IO.Path]::GetFullPath($root)
    $expected = [IO.Path]::GetFullPath((Join-Path $env:ProgramData 'GptBotCoverageAuthority'))
    if ($directoryCreated -and $resolved -eq $expected -and (Test-Path -LiteralPath $resolved)) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
    throw
}
