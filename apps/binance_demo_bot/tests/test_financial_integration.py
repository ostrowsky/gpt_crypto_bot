"""FRI-01..03: source-backed actual accounting, no network/credentials."""
from copy import deepcopy
from decimal import Decimal
import unittest

import _support
import test_financial_account as fixtures
from test_financial_account import current, balances, rehash


class FinancialIntegrationTests(unittest.TestCase):
    setUp=fixtures.FinancialAccountTests.setUp
    make=fixtures.FinancialAccountTests.make
    reconcile=fixtures.FinancialAccountTests.reconcile
    fill=fixtures.FinancialAccountTests.fill
    reason=fixtures.FinancialAccountTests.reason

    def fixture(self,**fill):
        self.fill(**fill)
        history=deepcopy(current()['metadata'])
        args=balances(current(),USDT='79',USDC='20',AAA='2')
        for key in ('request_started_at_ms','received_at_ms'):args['metadata'][key]+=9000
        for key in ('request_started_monotonic_ns','received_monotonic_ns'):args['metadata'][key]+=9_000_000_000
        return args,history

    def test_fri01_earlier_mapping_imports_actual_fill_with_fresh_current_catalog(self):
        args,history=self.fixture()
        result=self.reconcile(args,metadata_history=[history])
        self.assertEqual(result['fills']['imported_n'],1)
        self.assertEqual(result['balance_check'],'BALANCES_MATCH')
        self.assertEqual(next(iter(self.store._imports.values()))['recipe']['metadata_sha256'],history['source_sha256'])

    def test_fri01_future_only_mapping_never_imports(self):
        args,history=self.fixture()
        result=self.reconcile(args,metadata_history=[args['metadata']])
        self.assertEqual(result['fills']['unimported_n'],1)
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fri01_tampering_fails_before_any_import(self):
        args,history=self.fixture();history['data']['symbols'][0]['baseAsset']='BAD'
        self.reason('FINANCE_SOURCE_INVALID',self.reconcile,args,metadata_history=[history])
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fri01_foreign_or_future_observation_rejected_before_import(self):
        for key,value in [('environment','BINANCE_SPOT'),('received_at_ms',30000)]:
            args,history=self.fixture();history[key]=value
            self.reason('FINANCE_SOURCE_INVALID',self.reconcile,args,metadata_history=[history])
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fri01_latest_pre_fill_mapping_selected_deterministically(self):
        args,history=self.fixture();older=deepcopy(history)
        for key in ('request_started_at_ms','received_at_ms'):older[key]-=1000
        for key in ('request_started_monotonic_ns','received_monotonic_ns'):older[key]-=1_000_000_000
        older['data']['serverTime']=1;rehash(older)
        self.reconcile(args,metadata_history=[history,older])
        self.assertEqual(next(iter(self.store._imports.values()))['recipe']['metadata_sha256'],history['source_sha256'])

    def test_fri01_same_time_conflicting_mapping_rejected_before_import(self):
        args,history=self.fixture();other=deepcopy(history)
        next(r for r in other['data']['symbols'] if r['symbol']=='AAAUSDT')['baseAsset']='BAD';rehash(other)
        self.reason('FINANCE_METADATA_CONFLICT',self.reconcile,args,metadata_history=[history,other])
        self.assertEqual(len(self.store._ledger.events()),0)

    def test_fri01_restart_does_not_debit_again_or_reprice_historical_fee(self):
        args,history=self.fixture();self.reconcile(args,metadata_history=[history])
        self.store.close();self.store=self.make()
        result=self.reconcile(args,check_id='again')
        self.assertEqual(result['fills']['imported_n'],1)
        self.assertEqual(len(self.store._ledger.events()),1)
        self.assertEqual(self.store.projection({'USDT':'1','USDC':'1','AAA':'100'})['attribution']['fee_expenses_usdt'],1)

    def test_fri01_changed_history_cannot_reuse_check_id(self):
        args,history=self.fixture();self.reconcile(args,metadata_history=[history])
        self.reason('FINANCE_CHECK_CONFLICT',self.reconcile,args,metadata_history=[])

    def test_fri02_base_fee_equity_bridge_debits_once(self):
        args,history=self.fixture(fee='0.1',fee_asset='AAA')
        args=balances(args,USDT='80',USDC='20',AAA='1.9')
        self.reconcile(args,metadata_history=[history]);report=self.store.interval_report('check-1')
        self.assertEqual(report['attribution_status'],'MATCH')
        self.assertEqual(Decimal(report['attribution']['fee_expenses_usdt']),1)
        self.assertEqual(Decimal(report['projected_equity_change_usdt']),-1)
        self.assertEqual(Decimal(report['attribution_residual_usdt']),0)
        self.assertIsNone(report['daily_net_pnl_usdt'])

    def test_fri02_seed_usdc_revaluation_is_included(self):
        args=current();args['tickers']['data'][0].update(bidPrice='1.08',askPrice='1.10');rehash(args['tickers'])
        self.reconcile(args);report=self.store.interval_report('check-1')
        self.assertEqual(report['attribution_status'],'MATCH')
        self.assertEqual(Decimal(report['projected_equity_change_usdt']),2)
        self.assertEqual(Decimal(report['attribution']['unrealized_usdt']),2)
        self.assertIsNone(report['daily_net_pnl_usdt'])

    def test_fri02_balance_residual_blocks_financial_attribution(self):
        self.reconcile(balances(current(),USDT='101',USDC='20'))
        report=self.store.interval_report('check-1')
        self.assertEqual(report['attribution_status'],'BLOCKED')
        self.assertIsNone(report['projected_equity_change_usdt'])

    def test_fri02_missing_mark_blocks_instead_of_zero_valuation(self):
        self.reconcile(balances(current(),USDT='100',USDC='20',XYZ='1'))
        report=self.store.interval_report('check-1')
        self.assertEqual(report['attribution_status'],'BLOCKED')
        self.assertIsNone(report['observed_equity_usdt'])

    def test_fri03_stale_journal_prefix_cannot_reuse_attribution(self):
        self.reconcile();self.fill();self.reconcile(balances(current(check_id='second'),USDT='79',USDC='20',AAA='2'))
        report=self.store.interval_report('check-1')
        self.assertEqual(report['attribution_status'],'BLOCKED')
        self.assertIn('JOURNAL_CHANGED',report['reasons'])

    def test_fri03_matching_balances_never_certify_daily_boundaries(self):
        self.reconcile();report=self.store.interval_report('check-1')
        self.assertEqual(report['financial_status'],'PENDING')
        self.assertIsNone(report['daily_net_pnl_usdt'])
        self.assertIn('CALENDAR_BOUNDARIES_PENDING',report['reasons'])
        self.assertFalse(report['execution_authorized'])


if __name__=='__main__':unittest.main()
