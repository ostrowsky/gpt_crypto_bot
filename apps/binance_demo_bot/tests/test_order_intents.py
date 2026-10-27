"""P3-A contracts: synthetic app-owned data, no keys or exchange calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal, localcontext
import importlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.isolation import AppPaths


IDENTITY = dict(account_alias='demo-test', episode_id='episode-a', scope_id='scope-a')


def intent(**changes):
    value = dict(arm_id='baseline', decision_id='decision-a', action_seq=0,
        symbol='AAAUSDT', base_asset='AAA', quote_asset='USDT', side='BUY',
        quantity='2', limit_price='100', fee_reserve='1', take_profit='110', stop_loss='95',
        asof_ms=900, expires_at_ms=2000, market_snapshot_id='market-a', account_snapshot_id='account-a',
        policy_version='policy-v1', model_id=None, source_sha256='a'*64, risk_contract_id='risk-v1')
    value.update(changes)
    return value


class OrderIntentTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.order_intents')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures'
        base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve())
        self.store=self.make()

    def make(self, **kwargs):
        values=dict(IDENTITY,allocations={'baseline':{'USDT':'1000','AAA':'10'},
                                        'challenger':{'USDT':'100','AAA':'0'}})
        values.update(kwargs)
        result=self.api.OrderIntentJournal(self.paths,**values)
        self.addCleanup(result.close)
        return result

    def reserve(self, **changes):
        return self.store.reserve(intent(**changes),now_ms=1000)

    def reason(self, code, call, *args, **kwargs):
        with self.assertRaises(AdapterError) as caught:
            result=call(*args,**kwargs)
            if hasattr(result,'close'):self.addCleanup(result.close)
        self.assertEqual(caught.exception.reason,code)

    def annotate(self, state, **changes):
        values=dict(expected_version=state['state_version'],lease_epoch=self.store.lease_epoch,now_ms=1000)
        values.update(changes)
        return self.store.annotate_send_started(state['intent_id'],**values)

    def cancel(self, state, **changes):
        values=dict(expected_version=state['state_version'],lease_epoch=self.store.lease_epoch,reason='OPERATOR_CANCEL')
        values.update(changes)
        return self.store.cancel_unsent(state['intent_id'],**values)

    def test_p3a01_atomic_reservation_and_pending_outbox(self):
        state=self.reserve()
        self.assertEqual(state['state'],'RESERVED')
        self.assertEqual(state['reservation'],dict(asset='USDT',amount='201'))
        self.assertEqual(state['outbox_id'],state['intent_id'])
        self.assertFalse(state['execution_authorized'])
        self.assertEqual(state['intent']['market_snapshot_id'],'market-a')
        with closing(sqlite3.connect(self.paths.runtime_path('oms-intents.sqlite'))) as con:
            rows=[json.loads(r[0])['data'] for r in con.execute("SELECT payload_json FROM records WHERE kind='DECISION'")]
        reservations=[r for r in rows if r['operation']=='RESERVE']
        self.assertEqual(len(reservations),1)
        self.assertEqual(reservations[0]['state'],state)
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT'],
                         dict(allocated='1000',reserved='201',available='799'))

    def test_p3a02_canonical_retry_keeps_ids_and_count(self):
        first=self.reserve()
        second=self.reserve(quantity=Decimal('2.00'),limit_price='100.000',fee_reserve='1.0')
        self.assertEqual(first,second)
        self.assertEqual(self.store.summary()['intents_N'],1)
        self.assertRegex(first['client_order_id'],r'^bd_[0-9a-f]{32}$')

    def test_p3a02_conflicting_key_is_not_new_order(self):
        self.reserve()
        self.reason('OMS_INTENT_CONFLICT',self.reserve,quantity='3')
        self.assertEqual(self.store.summary()['intents_N'],1)

    def test_p3a02_distinct_actions_and_episode_ids(self):
        a=self.reserve();b=self.reserve(action_seq=1)
        self.assertNotEqual(a['client_order_id'],b['client_order_id'])
        self.store.close()
        other_paths=AppPaths(self.paths.root/'other')
        other=self.api.OrderIntentJournal(other_paths,**dict(IDENTITY,episode_id='episode-b'),
                                         allocations={'baseline':{'USDT':'1000'}})
        self.addCleanup(other.close)
        self.assertNotEqual(a['client_order_id'],other.reserve(intent(),now_ms=1000)['client_order_id'])

    def test_p3a03_buy_aggregate_budget_includes_fee(self):
        self.reserve(quantity='9',fee_reserve='99')
        self.reason('OMS_INSUFFICIENT_ALLOCATION',self.reserve,decision_id='b',quantity='0.01',fee_reserve='1')
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['available'],'1')

    def test_p3a03_other_arm_cannot_borrow(self):
        self.reason('OMS_INSUFFICIENT_ALLOCATION',self.reserve,arm_id='challenger')
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['available'],'1000')

    def test_p3a03_sell_quantity_and_base_fee_reserved(self):
        state=self.reserve(side='SELL',quantity='9',fee_reserve='1',take_profit=None,stop_loss=None)
        self.assertEqual(state['reservation'],dict(asset='AAA',amount='10'))
        self.reason('OMS_INSUFFICIENT_ALLOCATION',self.reserve,side='SELL',decision_id='b',
                    quantity='0.1',fee_reserve='0',take_profit=None,stop_loss=None)

    def test_p3a03_pending_buy_does_not_fund_sell(self):
        self.reserve(arm_id='challenger',quantity='0.5')
        self.reason('OMS_INSUFFICIENT_ALLOCATION',self.reserve,arm_id='challenger',decision_id='sell',
                    side='SELL',quantity='0.5',fee_reserve='0',take_profit=None,stop_loss=None)

    def test_p3a04_exact_schema_symbol_and_sides(self):
        for changes in (dict(extra='x'),dict(symbol='AAABTC'),dict(side='MARKET'),dict(quote_asset='BTC'),
                        dict(action_seq=True),dict(arm_id='missing'),dict(source_sha256='z'*64),
                        dict(policy_version='with spaces'),dict(model_id='x'*257)):
            with self.subTest(changes=changes):self.reason('OMS_INVALID',self.reserve,**changes)
        missing=intent();missing.pop('account_snapshot_id')
        self.reason('OMS_INVALID',self.store.reserve,missing,now_ms=1000)

    def test_p3a04_numeric_types_nonfinite_and_bounds(self):
        for value in (0,-1,True,1.5,'NaN','Infinity','0','-2','1e-19','1e19','1.12345678901234567890123456789'):
            with self.subTest(value=value):self.reason('OMS_INVALID',self.reserve,quantity=value)
        self.reason('OMS_INVALID',self.reserve,fee_reserve='-1')

    def test_p3a04_protection_ordering_and_sell_levels(self):
        for changes in (dict(stop_loss='100'),dict(take_profit='100'),dict(stop_loss=None),
                        dict(take_profit='0'),dict(side='SELL')):
            with self.subTest(changes=changes):self.reason('OMS_INVALID',self.reserve,**changes)

    def test_p3a04_future_expired_and_invalid_clock(self):
        self.reason('OMS_FUTURE_SOURCE',self.reserve,asof_ms=1001)
        self.reason('OMS_EXPIRED',self.reserve,expires_at_ms=1000)
        for value in (True,-1,1000.0):
            self.reason('OMS_INVALID',self.store.reserve,intent(),now_ms=value)

    def test_p3a04_independent_decimal_context(self):
        with localcontext() as ctx:
            ctx.prec=3
            state=self.reserve(quantity='1.23456789',limit_price='100.12345678',fee_reserve='0.00000001')
        # Independent rational oracle: 618046023931953971 / 5000000000000000.
        self.assertEqual(state['reservation']['amount'],'123.6092047863907942')

    def test_p3a05_cas_and_epoch_are_checked(self):
        state=self.reserve()
        self.reason('OMS_STALE_VERSION',self.annotate,state,expected_version=0)
        self.reason('OMS_STALE_EPOCH',self.annotate,state,lease_epoch=self.store.lease_epoch-1)
        self.assertEqual(self.store.get(state['intent_id'])['state'],'RESERVED')

    def test_p3a05_only_one_started_attempt_unknown_retains_reserve(self):
        state=self.reserve();started=self.annotate(state)
        self.assertEqual(started['state'],'SEND_STARTED')
        self.assertFalse(started['execution_authorized'])
        self.reason('OMS_STALE_VERSION',self.annotate,state)
        unknown=self.store.mark_unknown(started['intent_id'],expected_version=started['state_version'],
                                      lease_epoch=self.store.lease_epoch,reason='TRANSPORT_TIMEOUT')
        self.assertEqual(unknown['state'],'QUERY_REQUIRED')
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'201')
        self.reason('OMS_ILLEGAL_TRANSITION',self.cancel,unknown)
        self.reason('OMS_ILLEGAL_TRANSITION',self.annotate,unknown)

    def test_p3a05_attempt_at_expiry_stays_unsent(self):
        state=self.reserve()
        self.reason('OMS_EXPIRED',self.annotate,state,now_ms=2000)
        self.assertEqual(self.store.get(state['intent_id'])['state'],'RESERVED')

    def test_p3a06_unsent_cancel_tombstone_and_retry_after_expiry(self):
        state=self.reserve();canceled=self.cancel(state)
        self.assertEqual(canceled['state'],'CANCELED')
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'0')
        self.assertEqual(self.store.reserve(intent(),now_ms=9999),canceled)
        self.reason('OMS_ILLEGAL_TRANSITION',self.annotate,canceled)
        self.assertEqual(self.store.summary()['intents_N'],1)

    def test_p3a06_returned_objects_detached(self):
        state=self.reserve();state['intent']['quantity']='999'
        state['reservation']['amount']='0'
        self.assertEqual(self.store.get(state['intent_id'])['intent']['quantity'],'2')
        summary=self.store.summary();summary['budgets']['baseline']['USDT']['allocated']='999'
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['allocated'],'1000')

    def test_p3a07_reopen_recovers_started_and_fences_old_epoch(self):
        original=self.reserve();started=self.annotate(original);epoch=self.store.lease_epoch
        self.store.close();self.store=self.make()
        self.assertGreater(self.store.lease_epoch,epoch)
        recovered=self.store.get(started['intent_id'])
        self.assertEqual(recovered['state'],'QUERY_REQUIRED')
        self.assertEqual(self.store.summary()['unresolved_n'],1)
        self.assertEqual(self.store.summary()['intents_N'],1)
        self.assertEqual(self.reserve(),recovered)
        self.reason('OMS_STALE_EPOCH',self.cancel,recovered,lease_epoch=epoch)
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'201')

    def test_p3a07_kill_persists_blocks_buy_allows_sell(self):
        pending=self.reserve();self.store.kill_entries('OPERATOR_STOP')
        self.reason('OMS_ENTRY_KILLED',self.annotate,pending)
        self.reason('OMS_ENTRY_KILLED',self.reserve,decision_id='new-buy')
        self.assertEqual(self.reserve(),pending)
        sell=self.reserve(side='SELL',decision_id='exit',fee_reserve='0',take_profit=None,stop_loss=None)
        self.assertEqual(self.annotate(sell)['state'],'SEND_STARTED')
        self.store.close();self.store=self.make()
        self.assertTrue(self.store.summary()['entry_killed'])
        self.reason('OMS_ENTRY_KILLED',self.reserve,decision_id='another-buy')

    def test_p3a07_exclusive_writer_and_changed_identity_denied(self):
        self.reason('OMS_BUSY',self.make)
        self.store.close()
        self.reason('OMS_IDENTITY',self.make,account_alias='other')
        self.reason('OMS_IDENTITY',self.make,allocations={'baseline':{'USDT':'999'}})

    def test_p3a08_native_sql_abort_leaves_no_partial_reserve(self):
        self.store._journal._con.execute("CREATE TRIGGER test_abort BEFORE INSERT ON records WHEN "
            "json_extract(NEW.payload_json,'$.data.operation')='RESERVE' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with self.assertRaises(AdapterError):self.reserve()
        with self.assertRaises(AdapterError):self.reserve(decision_id='retry')
        self.store.close()
        with closing(sqlite3.connect(self.paths.runtime_path('oms-intents.sqlite'))) as con:
            con.execute('DROP TRIGGER test_abort')
            con.commit()
        self.store=self.make()
        self.assertEqual(self.store.summary()['intents_N'],0)
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['available'],'1000')

    def test_p3a08_unknown_commit_reopen_recovers_single_intent(self):
        def hook(phase):
            if phase=='after_commit':raise OSError('synthetic acknowledgement failure')
        self.store._journal.fault_hook=hook
        with self.assertRaises(AdapterError):self.reserve()
        with self.assertRaises(AdapterError):self.reserve()
        self.store._journal.fault_hook=None;self.store.close();self.store=self.make()
        recovered=self.reserve()
        self.assertEqual(recovered['state'],'RESERVED')
        self.assertEqual(self.store.summary()['intents_N'],1)
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'201')

    def test_p3a09_real_thread_budget_race_has_single_winner(self):
        def attempt(index):
            try:return self.reserve(decision_id=f'parallel-{index}',quantity='6')['state']
            except AdapterError as error:return error.reason
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(attempt,range(2)))
        self.assertCountEqual(results,['RESERVED','OMS_INSUFFICIENT_ALLOCATION'])
        self.assertEqual(self.store.summary()['intents_N'],1)

    def test_p3a09_tombstones_count_capacity(self):
        self.store.close();self.store=self.make(max_intents=1)
        first=self.reserve();self.cancel(first)
        self.reason('OMS_CAPACITY',self.reserve,decision_id='new')
        self.assertEqual(self.reserve()['state'],'CANCELED')

    def test_p3a09_malformed_semantic_event_rejected_on_reopen(self):
        self.store._journal.append('DECISION',dict(scope='OMS_INTENTS',operation='FAKE_FILLED',profit='999'))
        self.store.close()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a09_changed_payload_hash_rejected_on_reopen(self):
        self.reserve();self.store.close()
        with closing(sqlite3.connect(self.paths.runtime_path('oms-intents.sqlite'))) as con:
            names=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='records'")]
            for name in names:con.execute('DROP TRIGGER "'+name+'"')
            con.execute("UPDATE records SET payload_json=replace(payload_json,'201','202') "
                        "WHERE json_extract(payload_json,'$.data.operation')='RESERVE'")
            con.commit()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a10_block_decision_is_durable_without_new_intent(self):
        self.reason('OMS_INSUFFICIENT_ALLOCATION',self.reserve,quantity='20')
        self.store.close()
        with closing(sqlite3.connect(self.paths.runtime_path('oms-intents.sqlite'))) as con:
            rows=[json.loads(r[0])['data'] for r in con.execute("SELECT payload_json FROM records WHERE kind='DECISION'")]
        blocked=[r for r in rows if r['operation']=='BLOCK']
        self.assertEqual(len(blocked),1)
        self.assertEqual(blocked[0]['reason'],'OMS_INSUFFICIENT_ALLOCATION')

    def test_p3a10_no_execution_or_business_result_claim(self):
        result=self.store.summary()
        self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['authority'],'LOCAL_JOURNAL_ONLY')
        self.assertIsNone(result['unresolved_ratio'])
        for name in ('send','buy','sell','place_order','release_sent','record_fill'):
            self.assertFalse(hasattr(self.store,name))
        self.assertNotIn('net_pnl_usdt',result)
        self.assertNotIn('protected',result)

    def test_p3a09_boolean_cannot_replace_integer_version_in_valid_hash_event(self):
        from copy import deepcopy
        state=self.reserve()
        data=[row['data'] for row in self.store._journal.read(limit=100)
              if row['data'].get('operation')=='RESERVE'][0]
        # Different logical key; an otherwise legitimate, hash-chained event.
        forged=deepcopy(data)
        forged['state']['intent']['decision_id']='forged'
        forged['state']=self.store._initial(forged['state']['intent'])
        forged['state']['state_version']=True
        self.store._journal.append('DECISION',forged)
        self.store.close()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a09_boolean_epoch_is_not_integer_epoch(self):
        data=[row['data'] for row in self.store._journal.read(limit=100)
              if row['data'].get('operation')=='EPOCH'][0]
        # Epoch 1 represented as a bool is invalid even after genuine genesis.
        self.store._journal.append('DECISION',dict(data,epoch=True))
        self.store.close()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a09_preexisting_audit_without_genesis_does_not_mint_budget(self):
        from binance_demo_bot.interaction_audit import AuditJournal
        self.store.close()
        other=AppPaths(self.paths.root/'no-genesis')
        audit=AuditJournal(other,**IDENTITY,database='oms-intents.sqlite')
        audit.close()
        self.reason('OMS_INTEGRITY',self.api.OrderIntentJournal,other,**IDENTITY,
                    allocations={'baseline':{'USDT':'1000'}})

    def test_p3a10_idempotent_retry_is_audited_without_reserving_twice(self):
        state=self.reserve();self.reserve()
        rows=[r['data'] for r in self.store._journal.read(limit=100)
              if r['data'].get('operation')=='RETRY']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['intent_id'],state['intent_id'])
        self.assertEqual(rows[0]['state_version'],state['state_version'])
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'201')

    def test_p3a09_malformed_block_reference_rejected_on_reopen(self):
        self.store._journal.append('DECISION',dict(scope='OMS_INTENTS',operation='BLOCK',
            oms_source_sha256='a'*64,method='RESERVE',reason='OMS_INVALID',intent_id='not-an-id'))
        self.store.close()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a09_bool_epoch_on_reserve_is_not_valid_integer(self):
        state=self.store._initial(intent())
        self.store._journal.append('DECISION',dict(scope='OMS_INTENTS',operation='RESERVE',
            oms_source_sha256='a'*64,state=state,now_ms=1000,epoch=True))
        self.store.close()
        self.reason('OMS_INTEGRITY',self.make)

    def test_p3a08_uncertain_started_ack_recovers_without_second_attempt(self):
        state=self.reserve()
        def hook(phase):
            if phase=='after_commit':raise OSError('synthetic started ACK failure')
        self.store._journal.fault_hook=hook
        with self.assertRaises(AdapterError):self.annotate(state)
        self.store._journal.fault_hook=None;self.store.close();self.store=self.make()
        recovered=self.store.get(state['intent_id'])
        self.assertEqual(recovered['state'],'QUERY_REQUIRED')
        self.assertEqual(self.store.summary()['budgets']['baseline']['USDT']['reserved'],'201')
        self.reason('OMS_ILLEGAL_TRANSITION',self.annotate,recovered)

    def test_p3a08_failed_started_transaction_is_unsent_after_reopen(self):
        state=self.reserve()
        self.store._journal._con.execute("CREATE TRIGGER test_abort BEFORE INSERT ON records WHEN "
            "json_extract(NEW.payload_json,'$.data.operation')='SEND_STARTED' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with self.assertRaises(AdapterError):self.annotate(state)
        self.store.close()
        with closing(sqlite3.connect(self.paths.runtime_path('oms-intents.sqlite'))) as con:
            con.execute('DROP TRIGGER test_abort');con.commit()
        self.store=self.make()
        self.assertEqual(self.store.get(state['intent_id'])['state'],'RESERVED')

    def test_p3a07_actual_process_death_after_started_retains_unknown_reserve(self):
        root=self.paths.root/'process-death'
        source=Path(__file__).resolve().parents[1]/'src'
        script="""
import json, os, sys
sys.path.insert(0,sys.argv[1])
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.order_intents import OrderIntentJournal
store=OrderIntentJournal(AppPaths(sys.argv[2]),account_alias='demo-test',
    episode_id='episode-a',scope_id='scope-a',allocations={'baseline':{'USDT':'1000'}})
state=store.reserve(json.loads(sys.argv[3]),now_ms=1000)
store.annotate_send_started(state['intent_id'],expected_version=1,lease_epoch=store.lease_epoch,now_ms=1000)
os._exit(73)
"""
        result=subprocess.run([sys.executable,'-I','-c',script,str(source),str(root),json.dumps(intent())],
                              capture_output=True,timeout=30)
        self.assertEqual(result.returncode,73,result.stderr.decode(errors='replace'))
        reopened=self.api.OrderIntentJournal(AppPaths(root),**IDENTITY,allocations={'baseline':{'USDT':'1000'}})
        self.addCleanup(reopened.close)
        recovered=reopened.reserve(intent(),now_ms=1000)
        self.assertEqual(recovered['state'],'QUERY_REQUIRED')
        self.assertEqual(reopened.summary()['unresolved_ratio'],dict(n=1,N=1))
        self.assertEqual(reopened.summary()['budgets']['baseline']['USDT']['reserved'],'201')

    def test_p3a05_bool_cas_does_not_equal_numeric_version(self):
        state=self.reserve()
        self.reason('OMS_STALE_VERSION',self.annotate,state,expected_version=True)


if __name__=='__main__':unittest.main()
