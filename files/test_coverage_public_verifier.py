import base64
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import coverage_public_verifier as verifier


class PublicVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = {'contract': 'independent-market-policy-certification-v1', 'bundle_sha256':'test'}
        powershell = Path(os.environ.get('WINDIR',r'C:\Windows'))/'System32'/'WindowsPowerShell'/'v1.0'/'powershell.exe'
        if not powershell.exists():
            raise unittest.SkipTest('Windows cryptographic provider required for native integration test')
        # Ephemeral test key is never written or returned; only public key/signature.
        script = """
        $ErrorActionPreference='Stop'
        $rsa=New-Object Security.Cryptography.RSACng
        try {
            $rsa.KeySize=3072
            $data=[Convert]::FromBase64String([Console]::In.ReadToEnd())
            $sig=$rsa.SignData($data,[Security.Cryptography.HashAlgorithmName]::SHA256,
                [Security.Cryptography.RSASignaturePadding]::Pkcs1)
            @{public_xml=$rsa.ToXmlString($false); signature=[Convert]::ToBase64String($sig)} | ConvertTo-Json -Compress
        } finally { $rsa.Dispose() }
        """
        r=subprocess.run([str(powershell),'-NoProfile','-NonInteractive','-Command',script],
            input=base64.b64encode(verifier.canonical(cls.body)),capture_output=True,timeout=45,check=True)
        fixture=json.loads(r.stdout)
        raw=fixture['public_xml'].encode('ascii')
        cls.authority=verifier.PublicAuthority(raw,verifier.sha(raw))
        cls.cert={'contract':verifier.CONTRACT,'public_key_sha256':cls.authority.fingerprint,
                  'body':cls.body,'signature':fixture['signature']}

    def test_real_native_signature_and_changed_body(self):
        self.assertEqual(verifier.verify(self.cert,self.authority),self.body)
        with self.assertRaisesRegex(ValueError,'signature mismatch'):
            verifier.verify({**self.cert,'body':{**self.body,'bundle_sha256':'changed'}},self.authority)

    def test_missing_trust_store_does_not_fall_back_to_hmac(self):
        with self.assertRaisesRegex(ValueError,'trust store'):
            verifier.verify(self.cert,verifier.PublicAuthority(b'',''))

    def test_key_pin_mismatch_and_private_parameters_rejected(self):
        with self.assertRaises(ValueError):
            verifier.verify(self.cert,verifier.PublicAuthority(self.authority.raw,'wrong'))
        raw=self.authority.raw.replace(b'</RSAKeyValue>',b'<D>AA==</D></RSAKeyValue>')
        with self.assertRaisesRegex(ValueError,'public parameters'):
            verifier.verify(self.cert,verifier.PublicAuthority(raw,verifier.sha(raw)))

    def test_bad_signature_format_or_identity(self):
        for change in ({'signature':'bad!'}, {'public_key_sha256':'wrong'}, {'contract':'unknown'}):
            with self.assertRaises((ValueError,KeyError)):
                verifier.verify({**self.cert,**change},self.authority)

    def test_provider_failure_is_closed(self):
        with patch.object(verifier.subprocess,'run',side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(ValueError,'unavailable'):
                verifier.verify(self.cert,self.authority)

    def test_production_callers_use_public_material(self):
        for name in ('forward_evidence_service.py','independent_portfolio_confirmation.py','independent_portfolio_gate.py'):
            source=Path(verifier.__file__).with_name(name).read_text(encoding='utf-8')
            self.assertIn('authority_material()',source)
            self.assertNotIn("os.environ.get('RANKER_COVERAGE_AUTHORITY_KEY'",source)


if __name__ == '__main__': unittest.main()
