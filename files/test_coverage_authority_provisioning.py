"""Structural safety checks; not a substitute for Windows deployment verification."""
from pathlib import Path
import unittest


class ProvisioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).resolve().parents[1]/'install_coverage_authority.ps1').read_text()

    def test_restore_point_before_mutation_and_verified(self):
        self.assertLess(self.source.index('Checkpoint-Computer'), self.source.index('$created = $false'))
        self.assertIn("if (-not $point) { throw", self.source)

    def test_no_existing_principal_reuse(self):
        self.assertIn('refusing reuse or key replacement', self.source)

    def test_private_acl_excludes_evaluator_trainer(self):
        self.assertIn("Set-AuthorityAcl $private @{$sid='ReadAndExecute'}", self.source)
        self.assertIn("Set-AuthorityAcl $admin @{}", self.source)
        self.assertIn('SetAccessRuleProtection($true, $false)', self.source)

    def test_minimal_noninteractive_account(self):
        self.assertIn('SeBatchLogonRight', self.source)
        self.assertIn('SeDenyInteractiveLogonRight', self.source)
        self.assertIn('SeDenyRemoteInteractiveLogonRight', self.source)
        self.assertIn('Get-LocalGroup -SID S-1-5-32-545', self.source)
        self.assertNotIn('Add-LocalGroupMember -Group Administrators', self.source)

    def test_keys_not_exported_to_checkout_or_console(self):
        self.assertIn("$root = Join-Path $env:ProgramData $account", self.source)
        self.assertIn('$rsa.KeySize = 3072', self.source)
        self.assertIn('credential.dpapi', self.source)
        self.assertNotIn('Write-Output $password', self.source)
        self.assertNotIn('Write-Output $rsa', self.source)

    def test_no_claim_of_certification_or_activation(self):
        self.assertIn('PROVISIONED_NOT_CERTIFYING', self.source)
        self.assertIn('runtime_eligible=$false', self.source)
        self.assertIn('signing_task_installed=$false', self.source)

    def test_rollback_containment_verified(self):
        self.assertIn('$resolved -eq $expected', self.source)
        self.assertIn('if ($created) { Remove-LocalUser', self.source)


if __name__ == '__main__': unittest.main()
