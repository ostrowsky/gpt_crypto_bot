"""AUD-03 synthetic identity redaction before authenticated live diagnostics."""
import json
import unittest

import _support
from binance_demo_bot.adapter import DemoCredentials, HttpResponse
from binance_demo_bot.interaction_audit import response_payload, sanitize
import test_interaction_audit as fixtures


class IdentitySanitizerTests(unittest.TestCase):
    def test_uid01_numeric_account_identifier_removed_preserving_balance(self):
        payload=response_payload(b'{"uid":938475612,"balances":[{"asset":"USDT","free":"100"}]}',private=True)
        self.assertNotIn('uid',payload['data'])
        self.assertNotIn('938475612',json.dumps(payload))
        self.assertEqual(payload['data']['balances'][0]['free'],'100')

    def test_uid02_nested_encoded_case_and_text_identifiers_removed(self):
        value={'nested':[{'UID':938475613,'userId':'938475614','user_id':'938475615',
                         '%75id':'938475616','note':'uid=938475617'}]}
        saved=json.dumps(sanitize(value))
        for identifier in range(938475613,938475618):self.assertNotIn(str(identifier),saved)

    def test_uid03_identifier_in_public_response_precludes_exact_bytes(self):
        payload=response_payload(b'{"uid":938475618,"ok":true}',private=False)
        self.assertEqual(payload['representation'],'SANITIZED')
        self.assertNotIn('body_base64',payload)
        self.assertNotIn('938475618',json.dumps(payload))

    def test_uid04_opaque_namespace_and_economics_are_preserved(self):
        value={'account_identity':'a'*64,'account_alias':'demo-primary','quantity':'12.25',
               'symbol':'BTCUSDT','note':'liquidity and rapid quote updates'}
        self.assertEqual(sanitize(value),value)


class PersistedIdentityTests(unittest.TestCase):
    # Reuse only fixture/helpers, without discovering inherited test methods.
    setUp=fixtures.AuditCase.setUp
    make=fixtures.AuditCase.make
    rows=fixtures.AuditCase.rows
    client=fixtures.AuditCase.client
    def test_uid05_private_rest_receipt_is_redacted_before_durable_append(self):
        client=self.client([HttpResponse(200,b'{"uid":938475619,"balances":[]}',{})])
        client._credentials=DemoCredentials('synthetic-key','synthetic-secret')
        client._request('/api/v3/account',{},signed=True,weight=1)
        self.assertNotIn('938475619',json.dumps(self.rows()))
        self.journal.close();self.journal=self.make();self.addCleanup(self.journal.close)
        self.assertNotIn('938475619',json.dumps(self.rows()))


if __name__=='__main__':unittest.main()
