"""OPN-01..07: synthetic account/bootstrap evidence, no credentials or network."""
from copy import deepcopy
from decimal import Decimal, localcontext
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError, _sha
from binance_demo_bot.isolation import AppPaths

IDENTITY = dict(account_alias='test-demo', episode_id='opening-1', scope_id='whole-account')
HMAC = 'a'*64


def observation(data, endpoint, start, end, *, private=True):
    value = dict(data=data, source_sha256=_sha(data), source_method='GET',
        source_endpoint='/api/v3/'+endpoint, environment='BINANCE_SPOT_DEMO',
        request_started_at_ms=start, received_at_ms=end,
        request_started_monotonic_ns=start*1_000_000, received_monotonic_ns=end*1_000_000,
        clock_uncertainty_ms=10, audit_begin_event_id='b'*32, audit_end_event_id='c'*32)
    if private:
        value.update(schema_version=1, application_id='binance_demo_bot',
            objective_contract_id='daily_net_equity_pnl_v1', account_alias=IDENTITY['account_alias'],
            account_episode_id=IDENTITY['episode_id'], portfolio_scope_id=IDENTITY['scope_id'],
            credential_generation='generation-1')
    return value


def account(rows=None):
    rows = rows if rows is not None else [dict(asset='USDT', free='100', locked='0'),
        dict(asset='USDC', free='20', locked='0'), dict(asset='DUST', free='0', locked='0')]
    return dict(balances=rows, returned_assets=sorted(r['asset'] for r in rows),
        coverage='RETURNED_BALANCES_ONLY', account_identity=HMAC, can_trade=True, permissions=['SPOT'],
        update_time_ms=10)


def inputs():
    return dict(candidate_id='candidate-1', cutoff_ms=10_000,
        account_before=observation(account(), 'account', 9000, 9100),
        orders=observation([], 'openOrders', 9200, 9300),
        order_lists=observation([], 'openOrderList', 9400, 9500),
        metadata=observation(dict(symbols=[dict(symbol='USDCUSDT', baseAsset='USDC',
            quoteAsset='USDT', status='TRADING')]), 'exchangeInfo', 8000, 8100, private=False),
        tickers=observation([dict(symbol='USDCUSDT', bidPrice='0.98', askPrice='1.00')],
            'ticker/bookTicker', 9600, 9700, private=False),
        account_after=observation(account(), 'account', 9800, 9900))


def rehash(value):
    value['source_sha256'] = _sha(value['data'])


class OpeningTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.account_opening')
        base = Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures'
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temporary.cleanup)
        self.paths = AppPaths(Path(temporary.name).resolve())
        self.store = self.make()

    def make(self, **changes):
        args = dict(IDENTITY, **changes)
        store = self.api.AccountOpeningManifest(self.paths, **args)
        self.addCleanup(store.close)
        return store

    def attest(self, **changes):
        args = dict(account_identity=HMAC, declaration_id='human-exclusive-1', declared_at_ms=8500,
                    exclusive=True, now_ms=10_000)
        args.update(changes)
        return self.store.attest(**args)

    def capture(self, value=None):
        return self.store.capture(**(inputs() if value is None else value))

    def reason(self, expected, call, *args, **kwargs):
        with self.assertRaises(AdapterError) as caught:
            result = call(*args, **kwargs)
            if hasattr(result, 'close'): self.addCleanup(result.close)
        self.assertEqual(caught.exception.reason, expected)

    def test_opn01_explicit_attestation_persisted_and_idempotent(self):
        first = self.attest()
        self.assertEqual(first, self.attest(now_ms=11_000))
        self.assertTrue(first['exclusive'])
        self.assertEqual(first['account_identity'], HMAC)
        self.assertEqual(self.store.summary()['attestation'], first)

    def test_opn01_no_implicit_attestation(self):
        self.reason('OPENING_ATTESTATION_REQUIRED', self.capture)
        for value in (False, None, 1, 'true'):
            with self.subTest(value=value): self.reason('OPENING_INVALID', self.attest, exclusive=value)

    def test_opn01_identity_time_and_reference_validation(self):
        for change in (dict(account_identity='raw-id'), dict(declaration_id=''), dict(declaration_id='has spaces'),
                       dict(declared_at_ms=10_001), dict(declared_at_ms=True), dict(now_ms=-1)):
            with self.subTest(change=change): self.reason('OPENING_INVALID', self.attest, **change)

    def test_opn01_conflicting_declaration_is_not_replaced(self):
        self.attest()
        self.reason('OPENING_CONFLICT', self.attest, account_identity='d'*64)
        self.reason('OPENING_CONFLICT', self.attest, declaration_id='other')
        self.assertEqual(self.store.summary()['attestation']['account_identity'], HMAC)

    def test_opn02_source_hash_and_audit_required(self):
        self.attest()
        for name in ('account_before','account_after','orders','order_lists','metadata','tickers'):
            for field, value in (('source_sha256','f'*64),('audit_end_event_id',None)):
                args=inputs(); args[name][field]=value
                with self.subTest(name=name,field=field): self.reason('OPENING_SOURCE_INVALID',self.capture,args)
        self.assertEqual(self.store.summary()['candidates_n'],0)

    def test_opn02_private_namespace_and_generation(self):
        self.attest()
        for field,value in (('account_alias','foreign'),('account_episode_id','old'),
                            ('portfolio_scope_id','other'),('credential_generation','different'),
                            ('application_id','legacy'),('objective_contract_id','sales')):
            args=inputs(); args['account_after'][field]=value
            with self.subTest(field=field): self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn02_foreign_hmac_rejected(self):
        self.attest(); args=inputs(); args['account_after']['data']['account_identity']='d'*64
        rehash(args['account_after'])
        self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn02_wrong_venue_method_endpoint(self):
        self.attest()
        for field,value in (('environment','MAINNET'),('source_method','POST'),('source_endpoint','/api/v3/order')):
            args=inputs(); args['tickers'][field]=value
            with self.subTest(field=field): self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn02_receipt_quality_and_freshness(self):
        self.attest()
        for name,field,value in (('account_after','received_at_ms',10_001),
            ('orders','request_started_at_ms',9400),('orders','received_monotonic_ns',1),
            ('tickers','clock_uncertainty_ms',251),('metadata','received_at_ms',True)):
            args=inputs(); args[name][field]=value
            with self.subTest(name=name,field=field): self.reason('OPENING_SOURCE_INVALID',self.capture,args)
        args=inputs(); args['cutoff_ms']=50_000
        self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn02_adapter_decimal_clock_uncertainty_is_accepted(self):
        self.attest();args=inputs()
        for name in ('account_before','account_after','orders','order_lists','metadata','tickers'):
            args[name]['clock_uncertainty_ms']=Decimal('154.166800')
        result=self.capture(args)
        self.assertEqual(result['state'],'PROVISIONAL_OPENING')
        self.assertEqual(result['sources']['tickers']['clock_uncertainty_ms'],'154.1668')

    def test_opn02_decimal_clock_limits_nonfinite_and_bool_are_rejected(self):
        self.attest()
        for value in (Decimal('250.00001'),Decimal('NaN'),Decimal('Infinity'),True,None,'10'):
            args=inputs();args['tickers']['clock_uncertainty_ms']=value
            with self.subTest(value=value):self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn02_account_reads_bracket_other_sources(self):
        self.attest()
        for name in ('orders','order_lists','tickers'):
            args=inputs(); args[name]['request_started_at_ms']=9050
            args[name]['request_started_monotonic_ns']=9050_000_000
            with self.subTest(name=name): self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn03_invalid_balance_rows(self):
        self.attest()
        for rows in ([dict(asset='USDT',free='-1',locked='0')],
                     [dict(asset='USDT',free=1.0,locked='0')],
                     [dict(asset='USDT',free='NaN',locked='0')],
                     [dict(asset='USDT',free='1',locked='0')]*2):
            args=inputs(); args['account_before']['data']=account(rows); rehash(args['account_before'])
            with self.subTest(rows=rows): self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn03_returned_asset_metadata_not_silently_ignored(self):
        self.attest(); args=inputs(); args['account_before']['data']['returned_assets']=['USDT']
        rehash(args['account_before'])
        self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn03_unicode_returned_asset_not_dropped(self):
        self.attest();args=inputs()
        for name in ('account_before','account_after'):
            rows=args[name]['data']['balances']+[dict(asset='币安人生',free='0',locked='0')]
            args[name]['data']=account(rows);rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['state'],'PROVISIONAL_OPENING')
        self.assertEqual(result['returned_assets_N'],4)

    def test_opn04_unicode_symbol_and_asset_can_be_valued(self):
        self.attest();args=inputs()
        for name in ('account_before','account_after'):
            args[name]['data']=account([dict(asset='牛来',free='20',locked='0')]);rehash(args[name])
        args['metadata']['data']['symbols']=[dict(symbol='牛来USDT',baseAsset='牛来',quoteAsset='USDT',status='TRADING')]
        args['tickers']['data']=[dict(symbol='牛来USDT',bidPrice='2',askPrice='2')]
        for name in ('metadata','tickers'):rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['valuation']['equity_usdt'],'40')
        self.assertEqual(result['seed_lots'][0]['asset'],'牛来')

    def test_opn04_unknown_unicode_asset_blocks_without_folding(self):
        self.attest();args=inputs()
        for name in ('account_before','account_after'):
            args[name]['data']=account([dict(asset='ＵＳＤＴ',free='1',locked='0')]);rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertEqual(result['held_assets_N'],1)
        self.assertIn('ＵＳＤＴ:MISSING_ROUTE',result['valuation']['reason_codes'])

    def test_opn04_usdc_is_not_assumed_pegged(self):
        self.attest(); result=self.capture()
        self.assertEqual(result['valuation']['equity_usdt'],'119.8')
        self.assertEqual(result['valuation']['marks']['USDC'],'0.99')
        self.assertEqual(result['held_assets_N'],2)
        self.assertEqual(result['valuation']['time_basis'],'OBSERVED_RESPONSE')
        self.assertTrue(result['valuation']['exchange_quote_age_unknown'])

    def test_opn04_unknown_dust_is_counted_and_blocks(self):
        self.attest(); args=inputs()
        for name in ('account_before','account_after'):
            args[name]['data']['balances'][2]['free']='0.00000001'; rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertEqual(result['held_assets_N'],3)
        self.assertIsNone(result['valuation']['equity_usdt'])
        self.assertEqual(result['seed_lots'],[])
        self.assertIn('DUST:MISSING_ROUTE',result['valuation']['reason_codes'])

    def test_opn04_crossed_quote_and_missing_quote_block(self):
        self.attest()
        for candidate,rows in (('crossed',[dict(symbol='USDCUSDT',bidPrice='2',askPrice='1')]),('missing',[])):
            args=inputs(); args['candidate_id']=candidate; args['tickers']['data']=rows; rehash(args['tickers'])
            result=self.capture(args)
            self.assertEqual(result['state'],'BLOCKED'); self.assertIsNone(result['valuation']['equity_usdt'])

    def test_opn04_inverse_and_multileg_conversion(self):
        self.attest(); args=inputs()
        args['metadata']['data']['symbols']=[dict(symbol='USDTAAA',baseAsset='USDT',quoteAsset='AAA',status='TRADING'),
            dict(symbol='AAABBB',baseAsset='AAA',quoteAsset='BBB',status='TRADING')]
        args['tickers']['data']=[dict(symbol='USDTAAA',bidPrice='2',askPrice='2'),
            dict(symbol='AAABBB',bidPrice='4',askPrice='4')]
        for name in ('account_before','account_after'):
            args[name]['data']=account([dict(asset='BBB',free='8',locked='0')]); rehash(args[name])
        for name in ('metadata','tickers'):rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['valuation']['equity_usdt'],'1')
        self.assertEqual(result['valuation']['routes']['BBB'],[['AAABBB',True],['USDTAAA',True]])

    def test_opn04_invalid_metadata_and_duplicate_books_rejected(self):
        self.attest()
        for name in ('metadata','tickers'):
            args=inputs()
            rows=args[name]['data']['symbols'] if name=='metadata' else args[name]['data']
            rows.append(deepcopy(rows[0])); rehash(args[name])
            with self.subTest(name=name):self.reason('OPENING_SOURCE_INVALID',self.capture,args)

    def test_opn05_changing_account_balances_blocks(self):
        self.attest(); args=inputs(); args['account_after']['data']['balances'][0]['free']='99'
        rehash(args['account_after']); result=self.capture(args)
        self.assertEqual(result['state'],'BLOCKED'); self.assertIn('BALANCES_CHANGED',result['reasons'])
        self.assertEqual(result['seed_lots'],[])

    def test_opn05_locked_and_foreign_orders_block_without_adoption(self):
        self.attest()
        for case in ('lock','order','list','permission'):
            args=inputs(); args['candidate_id']=case
            if case=='lock':
                for name in ('account_before','account_after'):
                    args[name]['data']['balances'][0].update(free='99',locked='1'); rehash(args[name])
            elif case in ('order','list'):
                name='orders' if case=='order' else 'order_lists'
                args[name]['data']=[dict(symbol='AAAUSDT')]; rehash(args[name])
            else:
                args['account_after']['data']['can_trade']=False; rehash(args['account_after'])
            result=self.capture(args)
            with self.subTest(case=case):
                self.assertEqual(result['state'],'BLOCKED');self.assertEqual(result['seed_lots'],[])
                self.assertFalse(result['execution_authorized'])

    def test_opn06_seeds_are_capital_not_managed_position_or_profit(self):
        self.attest(); result=self.capture()
        self.assertEqual(result['state'],'PROVISIONAL_OPENING')
        lots={r['asset']:r for r in result['seed_lots']}
        self.assertEqual(lots['USDC'],dict(asset='USDC',quantity='20',basis_usdt='19.8',ownership='SEED'))
        self.assertEqual(result['managed_inventory'],{})
        self.assertFalse(result['exchange_reconciled']);self.assertFalse(result['financial_episode_opened'])
        self.assertFalse(result['execution_authorized']);self.assertIsNone(result['daily_net_pnl_usdt'])
        self.assertIn('CERTIFIED_CUT_PENDING',result['remaining_gates'])

    def test_opn06_fixed_decimal_context(self):
        self.attest()
        with localcontext() as context:
            context.prec=2
            result=self.capture()
        self.assertEqual(result['valuation']['equity_usdt'],'119.8')
        self.assertEqual(result['seed_lots'][0]['basis_usdt'],'19.8')

    def test_opn07_retry_conflict_and_detached_return(self):
        self.attest(); first=self.capture(); self.assertEqual(first,self.capture())
        first['seed_lots'].clear()
        self.assertEqual(len(self.capture()['seed_lots']),2)
        args=inputs();args['cutoff_ms']=10_001
        self.reason('OPENING_CONFLICT',self.capture,args)
        self.assertEqual(self.store.summary()['candidates_n'],1)

    def test_opn07_reopen_preserves_manifest(self):
        declaration=self.attest(); candidate=self.capture(); self.store.close()
        self.store=self.make()
        self.assertEqual(self.store.summary()['attestation'],declaration)
        self.assertEqual(self.capture(),candidate)
        self.assertEqual(self.store.summary()['candidates_n'],1)

    def test_opn07_namespace_reopen_is_rejected(self):
        self.attest();self.store.close()
        self.reason('OPENING_IDENTITY',self.make,episode_id='other')

    def test_opn07_uncertain_commit_poisons_cached_reads_and_recovers(self):
        self.attest()
        def fault(phase):
            if phase=='after_commit':raise OSError('synthetic')
        self.store._journal.fault_hook=fault
        self.reason('AUDIT_COMMIT_UNKNOWN',self.capture)
        self.reason('OPENING_UNAVAILABLE',self.capture)
        self.store._journal.fault_hook=None;self.store.close();self.store=self.make()
        self.assertEqual(self.store.summary()['candidates_n'],1)
        self.assertEqual(self.capture()['state'],'PROVISIONAL_OPENING')

    def test_opn07_precommit_failure_does_not_create_candidate(self):
        self.attest()
        def fault(phase):
            if phase=='before_commit':raise OSError('synthetic')
        self.store._journal.fault_hook=fault
        self.reason('AUDIT_WRITE_FAILED',self.capture)
        self.store._journal.fault_hook=None;self.store.close();self.store=self.make()
        self.assertEqual(self.store.summary()['candidates_n'],0)

    def test_opn02_metadata_and_quote_age_have_separate_bounds(self):
        self.attest()
        args=inputs();args['cutoff_ms']+=1_000_000
        for name in ('account_before','account_after','orders','order_lists','tickers'):
            for field in ('request_started_at_ms','received_at_ms'):args[name][field]+=1_000_000
            for field in ('request_started_monotonic_ns','received_monotonic_ns'):args[name][field]+=1_000_000_000_000
        self.reason('OPENING_SOURCE_INVALID',self.capture,args) # Only metadata is stale.
        args=inputs()
        args['account_before'].update(request_started_at_ms=1000,received_at_ms=2000,
            request_started_monotonic_ns=1_000_000_000,received_monotonic_ns=2_000_000_000)
        args['tickers'].update(request_started_at_ms=3000,received_at_ms=4000,
            request_started_monotonic_ns=3_000_000_000,received_monotonic_ns=4_000_000_000)
        self.reason('OPENING_SOURCE_INVALID',self.capture,args) # Bracketed, but quote is 6s old.

    def test_opn04_nontrading_route_not_used(self):
        self.attest();args=inputs();args['metadata']['data']['symbols'][0]['status']='BREAK'
        rehash(args['metadata']);result=self.capture(args)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertEqual(result['held_assets_N'],2)
        self.assertIsNone(result['valuation']['equity_usdt'])

    def test_opn04_equal_length_routes_selected_lexically(self):
        self.attest();args=inputs()
        # Both routes reach USDT in two legs; AAA precedes ZZZ regardless of input order.
        rows=[('USDCZZZ','USDC','ZZZ','2'),('ZZZUSDT','ZZZ','USDT','3'),
              ('USDCAAA','USDC','AAA','4'),('AAAUSDT','AAA','USDT','5')]
        args['metadata']['data']['symbols']=[dict(symbol=s,baseAsset=b,quoteAsset=q,status='TRADING') for s,b,q,p in rows]
        args['tickers']['data']=[dict(symbol=s,bidPrice=p,askPrice=p) for s,b,q,p in rows]
        for name in ('metadata','tickers'):rehash(args[name])
        result=self.capture(args)
        self.assertEqual(result['valuation']['routes']['USDC'],[['USDCAAA',False],['AAAUSDT',False]])
        self.assertEqual(result['valuation']['equity_usdt'],'500')

    def test_opn07_closed_and_live_writers_are_fenced(self):
        self.attest()
        self.reason('OPENING_BUSY',self.make)
        self.store.close()
        self.reason('OPENING_UNAVAILABLE',self.capture)

    def test_opn07_empty_history_does_not_silently_become_new_attestation(self):
        self.store.close()
        self.reason('OPENING_INTEGRITY',self.make)

    def test_opn07_candidate_capacity_is_bounded(self):
        self.attest()
        for i in range(100):
            args=inputs();args['candidate_id']='candidate-'+str(i);self.capture(args)
        args=inputs();args['candidate_id']='overflow'
        self.reason('OPENING_CAPACITY',self.capture,args)
        self.assertEqual(self.store.summary()['candidates_n'],100)

    def test_opn06_lower_precision_and_traps_cannot_round_starting_lots(self):
        self.attest();args=inputs()
        for name in ('account_before','account_after'):
            args[name]['data']['balances'][1]['free']='20.123456789012345678';rehash(args[name])
        with localcontext() as context:
            context.prec=1
            result=self.capture(args)
        self.assertEqual(result['seed_lots'][0]['basis_usdt'],'19.92222222112222222122')

    def test_opn07_persisted_source_pins_match_original_observations(self):
        self.attest();args=inputs();result=self.capture(args)
        for name in ('account_before','account_after','orders','order_lists','metadata','tickers'):
            self.assertEqual(result['sources'][name]['source_sha256'],args[name]['source_sha256'])
            self.assertEqual(result['sources'][name]['audit_end_event_id'],args[name]['audit_end_event_id'])
        result['sources'].clear()
        self.assertEqual(len(self.store.summary()['candidates']['candidate-1']['sources']),6)


if __name__=='__main__':unittest.main()
