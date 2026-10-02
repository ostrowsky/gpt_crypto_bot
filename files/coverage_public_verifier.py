"""Coverage signature verification using the Windows cryptographic provider only.

The evaluator never reads the authority private key. No keys are generated here.
"""
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from validated_ranker_rollout import canonical, sha

CONTRACT = 'coverage-rsa-sha256-v1'
VERIFY = """
$ErrorActionPreference = 'Stop'
try {
    $r = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $rsa = New-Object Security.Cryptography.RSACng
    try {
        $rsa.FromXmlString($r.public_xml)
        $ok = $rsa.VerifyData([Convert]::FromBase64String($r.message),
            [Convert]::FromBase64String($r.signature),
            [Security.Cryptography.HashAlgorithmName]::SHA256,
            [Security.Cryptography.RSASignaturePadding]::Pkcs1)
        if (-not $ok) { exit 1 }
        [Console]::Out.Write('VALID')
    } finally { $rsa.Dispose() }
} catch { exit 2 }
"""


@dataclass(frozen=True)
class PublicAuthority:
    raw: bytes
    fingerprint: str


def authority_material():
    """Fixed OS-protected trust store; request files cannot choose a public key."""
    root = Path(os.environ.get('ProgramData', r'C:\ProgramData'))/'GptBotCoverageAuthority'/'public'
    try:
        raw = (root/'verification_key.xml').read_bytes()
        installation = json.loads((root/'installation.json').read_text(encoding='utf-8-sig'))
        return PublicAuthority(raw, installation['public_key_sha256'].lower())
    except (OSError, ValueError, KeyError, TypeError):
        # Returned sentinel lets existing confirmation/controller report BLOCKED,
        # rather than crashing the scheduled status writer or using shared HMAC.
        return PublicAuthority(b'', '')


def verify(value, authority):
    if not isinstance(authority, PublicAuthority) or not authority.raw or sha(authority.raw) != authority.fingerprint:
        raise ValueError('coverage public trust store unavailable or changed')
    if len(authority.raw) > 16384 or b'<!' in authority.raw:
        raise ValueError('invalid public key document')
    root = ET.fromstring(authority.raw)
    if root.tag != 'RSAKeyValue' or [e.tag for e in root] != ['Modulus', 'Exponent']:
        raise ValueError('only RSA public parameters are permitted')
    modulus = base64.b64decode(root[0].text, validate=True)
    exponent = base64.b64decode(root[1].text, validate=True)
    if len(modulus) != 384 or int.from_bytes(exponent, 'big') != 65537:
        raise ValueError('unsupported coverage RSA key')
    if value.get('contract') != CONTRACT or value.get('public_key_sha256') != authority.fingerprint:
        raise ValueError('coverage signature identity mismatch')
    signature = base64.b64decode(value['signature'], validate=True)
    if len(signature) != len(modulus):
        raise ValueError('invalid coverage signature size')
    body = value['body']
    request = {'public_xml': authority.raw.decode('ascii'),
               'message': base64.b64encode(canonical(body)).decode('ascii'),
               'signature': value['signature']}
    powershell = Path(os.environ.get('WINDIR', r'C:\Windows'))/'System32'/'WindowsPowerShell'/'v1.0'/'powershell.exe'
    try:
        result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command', VERIFY],
                                input=json.dumps(request).encode(), capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError('coverage cryptographic verification unavailable') from exc
    if result.returncode != 0 or result.stdout.strip() != b'VALID':
        raise ValueError('coverage authority signature mismatch')
    return body
