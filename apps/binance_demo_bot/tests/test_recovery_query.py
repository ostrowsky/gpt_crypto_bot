"""REC-02: exact GET recovery diagnostics; synthetic transport and journals only."""
from types import SimpleNamespace
from urllib.parse import parse_qs
import importlib
import unittest

import _support
import test_adapter as adapter_fixture
import test_execution_pilot as journal_fixture


class RecoveryQueryTests(unittest.TestCase):
    setUp = adapter_fixture.AdapterCase.setUp
    make_adapter = adapter_fixture.AdapterCase.make_adapter
    reply = adapter_fixture.AdapterCase.reply
    assert_reason = adapter_fixture.AdapterCase.assert_reason

    def gateway(self, client=None):
        module = importlib.import_module('binance_demo_bot.execution_pilot')
        gateway = module.MatchingGateway(client or self.adapter)
        self.addCleanup(gateway.close)
        return gateway

    def test_rec02_error_code_is_optional_safe_numeric_metadata(self):
        error = self.api.AdapterError('READ_UNAVAILABLE')
        self.assertIsNone(error.api_code)
        self.assertEqual(str(error), 'READ_UNAVAILABLE')
        for value in (-2013, 0, 123):
            error = self.api.AdapterError('API_REJECTED', api_code=value)
            self.assertEqual(error.api_code, value)
        for value in ('-2013', True, -2013.0, {'code': -2013}):
            with self.subTest(value=value):
                error = self.api.AdapterError('API_REJECTED', api_code=value)
                self.assertIsNone(error.api_code)

    def test_rec02_adapter_attaches_code_without_exchange_message(self):
        self.reply({'code': -2013, 'msg': 'synthetic untrusted exchange message'}, status=400)
        error = self.assert_reason('API_REJECTED', self.adapter._request,
            '/api/v3/order', {'symbol': 'BTCUSDT', 'origClientOrderId': 'original-client'}, signed=True, weight=4)
        self.assertEqual(error.api_code, -2013)
        self.assertEqual(str(error), 'API_REJECTED')

    def test_rec02_only_exact_native_order_not_found_is_translated(self):
        self.reply({'code': -2013, 'msg': 'Order does not exist.'}, status=400)
        self.assert_reason('PILOT_ORDER_NOT_FOUND', self.gateway().query_action,
            'POST', '/api/v3/order', {'symbol': 'BTCUSDT', 'newClientOrderId': 'original-client'})
        method, path, query, _, _ = self.calls[0]
        self.assertEqual((method, path), ('GET', '/api/v3/order'))
        values = parse_qs(query)
        self.assertEqual(values['symbol'], ['BTCUSDT'])
        self.assertEqual(values['origClientOrderId'], ['original-client'])
        self.assertNotIn('newClientOrderId', values)

    def test_rec02_order_list_not_found_keeps_original_list_id(self):
        self.reply({'code': -2013}, status=400)
        self.assert_reason('PILOT_ORDER_NOT_FOUND', self.gateway().query_action,
            'POST', '/api/v3/orderList/oco', {'symbol': 'BTCUSDT', 'listClientOrderId': 'original-list'})
        method, path, query, _, _ = self.calls[0]
        self.assertEqual((method, path), ('GET', '/api/v3/orderList'))
        self.assertEqual(parse_qs(query)['origClientOrderId'], ['original-list'])

    def test_rec02_found_lookup_uses_owned_order_id_without_client_replacement(self):
        response = adapter_fixture.order(orderId=21)
        self.reply(response)
        result = self.gateway().query_action('DELETE', '/api/v3/order', {'symbol': 'BTCUSDT', 'orderId': 21})
        self.assertEqual(result, response)
        method, path, query, _, _ = self.calls[0]
        self.assertEqual((method, path), ('GET', '/api/v3/order'))
        values = parse_qs(query)
        self.assertEqual(values['symbol'], ['BTCUSDT'])
        self.assertEqual(values['orderId'], ['21'])
        self.assertNotIn('origClientOrderId', values)

    def test_rec02_native_generic_rejection_is_not_absence(self):
        self.reply({'code': -1100}, status=400)
        error = self.assert_reason('API_REJECTED', self.gateway().query_action,
            'POST', '/api/v3/order', {'symbol': 'BTCUSDT', 'newClientOrderId': 'original-client'})
        self.assertEqual(error.api_code, -1100)

    def test_rec02_read_failures_propagate_without_becoming_not_found(self):
        for reason in ('CLOCK_JUMP', 'CLOCK_QUALITY', 'AUTH_DENIED', 'SIGNATURE_REJECTED',
                       'RATE_LIMITED', 'AUDIT_WRITE_FAILED', 'READ_UNAVAILABLE', 'TRANSPORT_FAILURE', 'API_REJECTED'):
            with self.subTest(reason=reason):
                expected = self.api.AdapterError(reason)
                def request(*args, **kwargs):
                    raise expected
                error = self.assert_reason(reason, self.gateway(SimpleNamespace(_request=request)).query_action,
                    'POST', '/api/v3/order', {'symbol': 'BTCUSDT', 'newClientOrderId': 'original-client'})
                self.assertIs(error, expected)

    def test_rec02_not_found_code_does_not_override_other_failure_reason(self):
        expected = self.api.AdapterError('AUTH_DENIED', api_code=-2013)
        def request(*args, **kwargs):
            raise expected
        error = self.assert_reason('AUTH_DENIED', self.gateway(SimpleNamespace(_request=request)).query_action,
            'POST', '/api/v3/order', {'symbol': 'BTCUSDT', 'newClientOrderId': 'original-client'})
        self.assertIs(error, expected)


class RecoveryQueryJournalTests(unittest.TestCase):
    setUp = journal_fixture.PilotTests.setUp
    make = journal_fixture.PilotTests.make
    buy = journal_fixture.PilotTests.buy
    reason = journal_fixture.PilotTests.reason

    def test_rec02_not_found_preserves_unknown_and_prevents_resend(self):
        self.broker.error = TimeoutError()
        self.reason('PILOT_QUERY_REQUIRED', self.buy)
        from binance_demo_bot.adapter import AdapterError
        def not_found(*args):
            raise AdapterError('PILOT_ORDER_NOT_FOUND')
        self.broker.query_action = not_found
        self.reason('PILOT_ORDER_NOT_FOUND', self.book.recover, 'buy-1', self.broker)
        self.assertEqual(self.book.actions()['buy-1']['state'], 'QUERY_REQUIRED')
        self.assertEqual(self.book.owned_order_keys(), [])
        self.reason('PILOT_QUERY_REQUIRED', self.buy)
        self.assertEqual(len(self.broker.calls), 1)


if __name__ == '__main__':
    unittest.main()
