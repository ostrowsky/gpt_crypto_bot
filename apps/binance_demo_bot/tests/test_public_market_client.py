"""P2A-01 transport boundaries before refactoring the P1 read engine."""

import gzip
import importlib
import json
import unittest
from unittest.mock import patch

import _support


class PublicReadTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("binance_demo_bot.adapter")

    def test_P2A01_credential_free_client_rejects_private_before_dispatch(self):
        with patch.object(self.api.DemoHttpTransport, "request") as send:
            client = self.api.DemoReadClient()
            with self.assertRaises(self.api.AdapterError) as error:
                client.request_public("/api/v3/account", {}, weight=20)
            self.assertEqual(error.exception.reason, "FORBIDDEN_ENDPOINT")
            send.assert_not_called()

    def test_P2A01_no_credentials_on_public_request(self):
        outer = self
        class Transport:
            def request(inner, method, path, query, headers, timeout_s):
                outer.assertEqual(headers, {})
                outer.assertNotIn("signature", query)
                return outer.api.HttpResponse(200, b"{}")
        client = self.api.DemoReadClient(transport=Transport())
        result, meta = client.request_public("/api/v3/ping", {}, weight=1)
        self.assertEqual(result, {})
        import hashlib
        self.assertEqual(meta["raw_body_sha256"], hashlib.sha256(b"{}").hexdigest())

    def test_P2A01_metadata_large_reply_allowed_private_limit_unchanged(self):
        raw = json.dumps({"padding": "x"*(9*1024*1024)}).encode()
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = []
            response.read1.side_effect = [raw, b""]
            transport = self.api.DemoHttpTransport()
            result = transport.request("GET", "/api/v3/exchangeInfo", "", {}, 15)
            self.assertEqual(len(result.body), len(raw))
            transport.close()
            response.read1.side_effect = [raw, b""]
            with self.assertRaises(self.api.AdapterError) as error:
                transport.request("GET", "/api/v3/account", "", {}, 15)
            self.assertEqual(error.exception.reason, "RESPONSE_TOO_LARGE")

    def test_P2A01_public_gzip_bounded_and_corrupt_rejected(self):
        raw = json.dumps({"padding": "x"*(9*1024*1024)}).encode()
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = [("Content-Encoding", "gzip")]
            response.read1.side_effect = [gzip.compress(raw), b""]
            transport = self.api.DemoHttpTransport()
            self.assertEqual(transport.request("GET", "/api/v3/exchangeInfo", "", {}, 15).body, raw)
            transport.close()
            response.read1.side_effect = [b"not gzip", b""]
            with self.assertRaises(self.api.AdapterError):
                transport.request("GET", "/api/v3/exchangeInfo", "", {}, 15)

    def test_P2A01_decompressed_metadata_cap_prevents_zip_bomb(self):
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = [("Content-Encoding", "gzip")]
            response.read1.side_effect = [gzip.compress(b"x"*(32*1024*1024+1)), b""]
            with self.assertRaises(self.api.AdapterError) as error:
                self.api.DemoHttpTransport().request("GET", "/api/v3/exchangeInfo", "", {}, 15)
            self.assertEqual(error.exception.reason, "RESPONSE_TOO_LARGE")

    def test_P2A01_gzip_advertised_only_for_metadata(self):
        captured = []
        class Transport:
            def request(inner, method, path, query, headers, timeout_s):
                captured.append(headers)
                return self.api.HttpResponse(200, b"{}")
        self.api.DemoReadClient(transport=Transport()).request_public("/api/v3/exchangeInfo", {}, weight=20)
        self.assertEqual(captured, [{"Accept-Encoding": "gzip"}])

    def test_P2A01_transport_gzip_header_allowlist_is_metadata_only(self):
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = []
            response.read1.side_effect = [b"{}", b""]
            transport = self.api.DemoHttpTransport()
            self.assertEqual(transport.request("GET", "/api/v3/exchangeInfo", "", {"Accept-Encoding":"gzip"}, 15).body, b"{}")
            with self.assertRaises(self.api.AdapterError):
                transport.request("GET", "/api/v3/account", "", {"Accept-Encoding":"gzip"}, 15)
            transport.close()
