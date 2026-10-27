"""FAC-01..07; actual own SQLite, synthetic exchange facts, no credentials/network."""
from copy import deepcopy
from decimal import Decimal, localcontext
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _support
from test_account_opening import IDENTITY,HMAC,inputs,account,rehash
from _p1_support import normalized_fill,normalized_event,normalized_order
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.account_opening import AccountOpeningManifest
from binance_demo_bot.account_evidence import AccountEvidenceStore
from binance_demo_bot.isolation import AppPaths


def current(**changes):
    args=inputs();args.pop('candidate_id');args['check_id']='check-1';args['cutoff_ms']=20000
    for name in ('account_before','account_after','orders','order_lists','tickers'):
        for key in ('request_started_at_ms','received_at_ms'):args[name][key]+=10000
        for key in ('request_started_monotonic_ns','received_monotonic_ns'):args[name][key]+=10_000_000_000
    args['metadata']['data']['symbols'].append(dict(symbol='AAAUSDT',baseAsset='AAA',quoteAsset='USDT',status='TRADING'))
    args['tickers']['data'].append(dict(symbol='AAAUSDT',bidPrice='10',askPrice='10'))
    for name in ('metadata','tickers'):rehash(args[name])
    args.update(changes)
    return args


def balances(args,**amounts):
    rows=[dict(asset=a,free=q,locked='0') for a,q in amounts.items()]
    for name in ('account_before','account_after'):
        args[name]['data']=account(deepcopy(rows));rehash(args[name])
    return args


class FinancialAccountTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.financial_account')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temp.cleanup)
        self.paths=AppPaths(Path(temp.name).resolve())
        self.opening=AccountOpeningManifest(self.paths,**IDENTITY);self.addCleanup(self.opening.close)
        self.opening.attest(account_identity=HMAC,declaration_id='human-1',declared_at_ms=8500,exclusive=True,now_ms=10000)
        self.seed=self.opening.capture(**inputs())
        self.evidence=AccountEvidenceStore(self.paths,**IDENTITY,account_identity=HMAC);self.addCleanup(self.evidence.close)
        self.evidence.bind_session('session-1',0,'generation-1')
        self.store=self.make()

    def make(self,**changes):
        args=dict(opening_manifest=self.opening,candidate_id='candidate-1');args.update(changes)
        result=self.api.FinancialAccount(self.paths,**args);self.addCleanup(result.close)
        return result

    def reconcile(self,args=None,**changes):
        args=deepcopy(current() if args is None else args);args.update(changes)
        return self.store.reconcile(evidence=self.evidence,**args)

    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:
            result=call(*args,**kwargs)
            if hasattr(result,'close'):self.addCleanup(result.close)
        self.assertEqual(caught.exception.reason,code)

    def fill(self,trade=1,*,at=12000,received=12500,fee='1',fee_asset='USDT',side='BUY',
             quantity='2',price='10',quote='20',symbol='AAAUSDT'):
        payload=normalized_fill(trade,symbol=symbol,order_id=100+trade,side=side,
            price=price,quantity=quantity,quote_quantity=quote,commission_quantity=fee,
            commission_asset=fee_asset,exchange_event_time_ms=at)
        order=normalized_order(symbol=symbol,order_id=100+trade,side=side,status='FILLED',
            price=price,original_quantity=quantity,executed_quantity=quantity,cumulative_quote_quantity=quote,
            time_ms=at,update_time_ms=at)
        event=normalized_event(fill=payload,execution_id=trade,exchange_event_time_ms=at,
            transaction_time_ms=at,order=order)
        return self.evidence.record_event('session-1',0,event,received,received*1_000_000,'original-process')

    def test_fac01_original_seed_and_basis_bound_once(self):
        projection=self.store.projection({'USDT':'1','USDC':'0.99'})
        self.assertEqual(projection['balances'],{'USDC':Decimal('20'),'USDT':Decimal('100')})
        self.assertTrue(all(v==0 for v in projection['managed_quantities'].values()))
        self.assertEqual(self.store.summary()['opening_sha256'],self.seed['manifest_sha256'])

    def test_fac01_rebase_candidate_is_rejected(self):
        args=inputs();args['candidate_id']='other';args['cutoff_ms']=10001
        self.opening.capture(**args);self.store.close()
        self.reason('FINANCE_BINDING_CONFLICT',self.make,candidate_id='other')

    def test_fac01_blocked_seed_cannot_initialize(self):
        args=inputs();args['candidate_id']='blocked';args['account_after']['data']['can_trade']=False
        rehash(args['account_after']);self.opening.capture(**args);self.store.close()
        self.reason('FINANCE_OPENING_INVALID',self.make,candidate_id='blocked')

    def test_fac01_reopen_retains_frozen_seed(self):
        self.reconcile();self.store.close();self.store=self.make()
        self.assertEqual(self.store.summary()['checks_n'],1)
        self.assertEqual(self.store.projection({'USDT':'1','USDC':'1'})['balances']['USDT'],100)

    def test_fac02_seed_snapshot_cannot_validate_itself(self):
        args=inputs();args.pop('candidate_id');args['check_id']='same'
        self.reason('FINANCE_SOURCE_INVALID',self.reconcile,args)

    def test_fac02_invalid_source_never_imports_fill(self):
        self.fill();args=current();args['account_after']['source_sha256']='f'*64
        self.reason('FINANCE_SOURCE_INVALID',self.reconcile,args)
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fac02_stale_and_mixed_generation_rejected(self):
        for args in (current(cutoff_ms=60000),current()):
            args['account_after']['credential_generation']='different'
            self.reason('FINANCE_SOURCE_INVALID',self.reconcile,args)

    def test_fac03_independent_balances_match_but_finance_pending(self):
        result=self.reconcile()
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual((result['matched_assets_n'],result['assets_N']),(2,2))
        self.assertEqual(result['status'],'PENDING')
        self.assertFalse(result['exchange_reconciled']);self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['financial_episode_opened']);self.assertIsNone(result['daily_net_pnl_usdt'])

    def test_fac03_gain_or_loss_residual_not_profit_or_reset(self):
        for i,quantity in enumerate(('101','99')):
            result=self.reconcile(balances(current(check_id='delta-'+str(i)),USDT=quantity,USDC='20'))
            self.assertEqual(result['balance_check'],'BALANCES_DIFFER')
            self.assertEqual(result['status'],'MISMATCH')
            self.assertIsNone(result['daily_net_pnl_usdt'])
            self.assertEqual(self.store.projection({'USDT':'1','USDC':'1'})['balances']['USDT'],100)

    def test_fac03_omitted_asset_is_unknown(self):
        result=self.reconcile(balances(current(),USDT='100'))
        row=next(r for r in result['assets'] if r['asset']=='USDC')
        self.assertIsNone(row['exchange_quantity']);self.assertEqual(row['reason'],'ASSET_COVERAGE_MISSING')
        self.assertEqual(result['balance_check'],'UNKNOWN')

    def test_fac03_explicit_returned_zero_is_real_mismatch(self):
        result=self.reconcile(balances(current(),USDT='100',USDC='0'))
        row=next(r for r in result['assets'] if r['asset']=='USDC')
        self.assertEqual(row['exchange_quantity'],'0');self.assertEqual(row['delta'],'-20')
        self.assertEqual(result['balance_check'],'BALANCES_DIFFER')

    def test_fac03_new_unicode_dust_not_ignored(self):
        result=self.reconcile(balances(current(),USDT='100',USDC='20',牛来='0.00000001'))
        self.assertEqual(result['assets_N'],3)
        self.assertIsNone(result['observed_equity_usdt'])
        self.assertIn('VALUATION_INCOMPLETE',result['reasons'])

    def test_fac03_zero_asset_named_token_preserves_membership_safely(self):
        from binance_demo_bot.adapter import _sha
        result=self.reconcile(balances(current(),USDT='100',USDC='20',TOKEN='0',COOKIE='0'))
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual(result['returned_assets_N'],4)
        self.assertIn(_sha('TOKEN'),result['returned_asset_sha256'])
        self.assertIn(_sha('COOKIE'),result['returned_asset_sha256'])

    def test_fac03_locks_and_orders_are_preserved_as_blocks(self):
        args=current()
        for name in ('account_before','account_after'):
            args[name]['data']['balances'][0].update(free='99',locked='1');rehash(args[name])
        args['orders']['data']=[{'client_order_id':'manual-order'}];rehash(args['orders'])
        result=self.reconcile(args)
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertIn('LOCKED_INVENTORY',result['reasons']);self.assertIn('OPEN_ORDERS_PRESENT',result['reasons'])

    def test_fac04_actual_evidence_fill_and_fee_imported_once(self):
        self.fill();args=balances(current(),USDT='79',USDC='20',AAA='2')
        result=self.reconcile(args)
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual(len(self.store._ledger.events()),1)
        self.assertEqual(self.store._ledger.events()[0]['payload']['ownership'],'UNATTRIBUTED')
        receipt=self.store._ledger.receipts()[0]['envelope']
        self.assertEqual(receipt['received_at_ms'],12500);self.assertEqual(receipt['process_instance_id'],'original-process')
        self.fill(received=12501)
        result=self.reconcile(args,check_id='again')
        self.assertEqual(len(self.store._ledger.events()),1)
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')

    def test_fac04_base_fee_uses_transaction_price(self):
        self.fill(fee='0.1',fee_asset='AAA')
        result=self.reconcile(balances(current(),USDT='80',USDC='20',AAA='1.9'))
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual(self.store._ledger.events()[0]['marks']['AAA'],'10')
        self.assertEqual(self.store.projection({'USDT':'1','USDC':'0.99','AAA':'10'})['managed_quantities']['AAA'],0)

    def test_fac04_chronological_fills_not_receipt_order(self):
        self.fill(2,at=14000,received=14500,side='SELL',quantity='1',price='12',quote='12',fee='0.2')
        self.fill(1,at=12000,received=15000)
        result=self.reconcile(balances(current(),USDT='90.8',USDC='20',AAA='1'))
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual(len(self.store._ledger.events()),2)

    def test_fac05_unknown_third_fee_does_not_use_current_price(self):
        self.fill(fee='0.1',fee_asset='BNB')
        result=self.reconcile()
        self.assertEqual(len(self.store._ledger.events()),0)
        self.assertEqual(result['fills']['unimported_n'],1)
        self.assertIn('HISTORICAL_FEE_MARK_MISSING',result['reasons'])

    def test_fac05_metadata_received_after_fill_is_not_historical_proof(self):
        self.fill();args=current();args['metadata'].update(request_started_at_ms=13000,
            received_at_ms=13100,request_started_monotonic_ns=13_000_000_000,received_monotonic_ns=13_100_000_000)
        result=self.reconcile(args)
        self.assertEqual(result['fills']['unimported_n'],1)
        self.assertIn('HISTORICAL_SYMBOL_MAPPING_MISSING',result['reasons'])

    def test_fac05_previously_imported_fill_keeps_original_historical_mapping(self):
        self.fill();args=balances(current(),USDT='79',USDC='20',AAA='2');self.reconcile(args)
        args['check_id']='refreshed'
        args['metadata'].update(request_started_at_ms=13000,received_at_ms=13100,
            request_started_monotonic_ns=13_000_000_000,received_monotonic_ns=13_100_000_000)
        result=self.reconcile(args)
        self.assertEqual(result['fills']['imported_n'],1)
        self.assertEqual(result['fills']['unimported_n'],0)
        self.assertNotIn('HISTORICAL_SYMBOL_MAPPING_MISSING',result['reasons'])
        self.assertEqual(len(self.store._ledger.events()),1)

    def test_fac05_non_usdt_quote_stays_unimported(self):
        self.fill(symbol='AAAEUR')
        args=current();args['metadata']['data']['symbols'].append(
            dict(symbol='AAAEUR',baseAsset='AAA',quoteAsset='EUR',status='TRADING'))
        rehash(args['metadata']);result=self.reconcile(args)
        self.assertIn('NON_USDT_FILL_UNSUPPORTED',result['reasons'])
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fac05_bad_fill_arithmetic_not_rewritten(self):
        self.fill(quote='21');result=self.reconcile()
        self.assertIn('ACCOUNTING_REJECTED',result['reasons'])
        self.assertEqual(result['fills']['unimported_n'],1)
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fac05_pre_anchor_fill_excluded_without_certification(self):
        self.fill(at=9000,received=12500)
        result=self.reconcile()
        self.assertEqual(result['fills']['pre_anchor_n'],1);self.assertEqual(len(self.store._ledger.events()),0)

    def test_fac05_future_source_receipt_is_deferred(self):
        self.fill(received=21000)
        result=self.reconcile()
        self.assertEqual(result['fills']['deferred_n'],1);self.assertEqual(len(self.store._ledger.events()),0)

    def test_fac06_check_retry_detached_and_conflict(self):
        result=self.reconcile();same=self.reconcile();self.assertEqual(result,same)
        same['assets'].clear();self.assertEqual(len(self.reconcile()['assets']),2)
        self.reason('FINANCE_CHECK_CONFLICT',self.reconcile,current(cutoff_ms=20001))

    def test_fac06_missing_ready_ledger_cannot_reseed(self):
        path=self.store._ledger.database_path;self.store.close();path.unlink()
        self.reason('FINANCE_LEDGER_MISSING',self.make)

    def test_fac06_truncated_ready_ledger_cannot_silently_initialize_seed(self):
        self.fill();self.reconcile(balances(current(),USDT='79',USDC='20',AAA='2'))
        path=self.store._ledger.database_path;self.store.close();path.write_bytes(b'')
        self.reason('FINANCE_INTEGRITY',self.make)
        self.assertEqual(path.stat().st_size,0)

    def test_fac06_live_writer_and_closed_coordinator_fenced(self):
        self.reason('FINANCE_BUSY',self.make)
        self.store.close();self.reason('FINANCE_UNAVAILABLE',self.reconcile)

    def test_fac06_after_commit_unknown_recovers_one_fill(self):
        self.fill()
        def fault(phase):
            if phase=='after_commit':raise OSError('synthetic')
        self.store._ledger.fault_hook=fault
        self.reason('FINANCE_JOURNAL_UNKNOWN',self.reconcile)
        self.reason('FINANCE_UNAVAILABLE',self.reconcile)
        self.store._ledger.fault_hook=None;self.store.close();self.store=self.make()
        result=self.reconcile(balances(current(),USDT='79',USDC='20',AAA='2'))
        self.assertEqual(result['balance_check'],'BALANCES_MATCH');self.assertEqual(len(self.store._ledger.events()),1)

    def test_fac06_before_commit_failure_has_no_economic_write(self):
        self.fill()
        def fault(phase):
            if phase=='before_commit':raise OSError('synthetic')
        self.store._ledger.fault_hook=fault
        self.reason('FINANCE_JOURNAL_UNKNOWN',self.reconcile)
        self.assertEqual(len(self.store._ledger.events()),0)
        self.store._ledger.fault_hook=None;self.store.close();self.store=self.make()
        self.assertEqual(len(self.store._ledger.events()),1)

    def test_fac06_own_audit_failure_blocks_cached_retry(self):
        self.reconcile()
        def fault(phase):
            if phase=='before_commit':raise OSError('synthetic')
        self.store._audit.fault_hook=fault
        self.reason('AUDIT_WRITE_FAILED',self.reconcile,current(check_id='next'))
        self.reason('FINANCE_UNAVAILABLE',self.reconcile)

    def test_fac06_interrupted_binding_before_ledger_creation_resumes(self):
        count=0
        def fault(phase):
            nonlocal count
            if phase=='after_commit':
                count+=1
                if count==2:raise OSError('synthetic')
        self.reason('AUDIT_COMMIT_UNKNOWN',self.make,database='interrupted-audit.sqlite',
            ledger_database='interrupted-ledger.sqlite',fault_hook=fault)
        resumed=self.make(database='interrupted-audit.sqlite',ledger_database='interrupted-ledger.sqlite')
        self.assertEqual(len(resumed._ledger.events()),0)
        self.assertEqual(resumed.summary()['opening_sha256'],self.seed['manifest_sha256'])

    def test_fac06_unknown_check_commit_recovered_without_duplicate_check(self):
        def fault(phase):
            if phase=='after_commit':raise OSError('synthetic')
        self.store._audit.fault_hook=fault
        self.reason('AUDIT_COMMIT_UNKNOWN',self.reconcile)
        self.store._audit.fault_hook=None;self.store.close();self.store=self.make()
        self.assertEqual(self.store.summary()['checks_n'],1)
        self.assertEqual(self.reconcile()['balance_check'],'BALANCES_MATCH')

    def test_fac07_incidents_never_cleared_by_matching_balances(self):
        self.evidence.record_incident('STREAM_GAP')
        result=self.reconcile()
        self.assertIn('STREAM_GAP',result['reasons'])
        self.assertEqual(len(self.evidence.incidents()),1)
        self.assertEqual(result['status'],'PENDING')

    def test_fac07_flow_event_not_automatically_profit_or_classified_transfer(self):
        event=dict(kind='FLOW',exchange_event_time_ms=12000,transaction_time_ms=12000,asset='USDT',delta='5')
        self.evidence.record_event('session-1',0,event,12500,12_500_000_000,'original-process')
        result=self.reconcile(balances(current(),USDT='105',USDC='20'))
        self.assertIn('UNCLASSIFIED_FLOW',result['reasons'])
        self.assertEqual(result['status'],'MISMATCH');self.assertIsNone(result['daily_net_pnl_usdt'])

    def test_fac07_changed_evidence_cannot_reuse_same_check_id(self):
        self.reconcile();self.evidence.record_incident('STREAM_GAP')
        self.reason('FINANCE_CHECK_CONFLICT',self.reconcile)

    def test_fac07_evidence_mutation_marks_comparison_stale(self):
        original=self.store._ledger.projection
        def changed(*args,**kwargs):
            self.evidence.record_incident('STREAM_GAP')
            return original(*args,**kwargs)
        with patch.object(self.store._ledger,'projection',side_effect=changed):result=self.reconcile()
        self.assertIn('EVIDENCE_CHANGED',result['reasons'])
        self.assertFalse(result['exchange_reconciled'])

    def test_fac07_equity_change_diagnostic_not_daily_reward(self):
        args=current();args['tickers']['data'][0].update(bidPrice='1.08',askPrice='1.10');rehash(args['tickers'])
        with localcontext() as context:
            context.prec=2
            result=self.reconcile(args)
        self.assertEqual(result['unadjusted_equity_change_usdt'],'2')
        self.assertIsNone(result['daily_net_pnl_usdt'])

    def test_fac07_foreign_evidence_identity_rejected(self):
        other=AccountEvidenceStore(self.paths,**IDENTITY,account_identity='c'*64,database='foreign.sqlite')
        self.addCleanup(other.close)
        self.reason('FINANCE_EVIDENCE_INVALID',self.store.reconcile,evidence=other,**current())


if __name__=='__main__':unittest.main()
