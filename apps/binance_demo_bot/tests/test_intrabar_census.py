"""IC01/05/06: full binding population and causal first-touch diagnostics."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _support
from binance_demo_bot.intrabar_census import bind_cases, first_touch, census_counts, collector_registration
from binance_demo_bot.intrabar_probe import collect_probe_run


def bar(at=0,o='100',h='101',l='99',c='100',step=60000):
    return [at,o,h,l,c,'1',at+step-1,'100',1,'0','0','0']


def example():
    task=dict(id='TARGET_FIRST-test-base-GRU',order='TARGET_FIRST',segment='test',cost='base',label='GRU')
    detail=dict(state='COMPLETE',ambiguous_executions_n=1,
        ambiguities=[dict(symbol='牛USDT',origin_ms=900000,stop=99.5,target=100.49999999999999,chosen_reason='TP')],
        trades=[dict(symbol='牛USDT',entry_ms=900000,exit_ms=900000,entry_price=100.,reason='TP')])
    return task,detail


class CensusTests(unittest.TestCase):
    def test_IC04_registration_drives_real_collector_report_contract(self):
        original=bar(h='101',l='99');original[5]='15';original[7]='1500';original[8]=15;original[6]=899999
        binding=dict(windows=[dict(symbol='AAAUSDT',origin_ms=0,row=original,seconds_required=False)],
                     truth_harness_at_registration=dict(status='FAIL'))
        prepared=collector_registration(binding)
        class Client:
            def request_public(self,path,parameters,*,weight):
                return [bar(i*60000) for i in range(15)],dict(received_at_ms=1000000)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime') as folder:
            # Only filesystem parent validation is isolated; the collector,
            # candle validator, report construction and serialization are real.
            with patch('binance_demo_bot.intrabar_probe._validate_binding',return_value=prepared):
                report=collect_probe_run(folder,client=Client(),include_seconds=False,max_attempts=3)
        self.assertEqual(report['state'],'MINUTES_ONLY');self.assertEqual(report['match_n'],1)
        self.assertEqual(report['rounding_contract'],'ASSUMED_NEAREST_ROUNDING_CAPPED_ORIGINAL_QUANTUM_1e-8')
        self.assertNotIn('rounding_contract',binding)

    def test_IC01_all_differing_bindings_preserved_and_no_return_filter(self):
        a,d=example(); cases=bind_cases(a,d)
        b=copy.deepcopy(a); b.update(id='STOP_FIRST-test-base-GRU',order='STOP_FIRST')
        e=copy.deepcopy(d);e['ambiguities'][0]['chosen_reason']='SL'; e['trades'][0]['reason']='SL'
        others=bind_cases(b,e)
        self.assertEqual(len(cases+others),2)
        self.assertEqual(cases[0]['stop'],99.5)
        self.assertEqual(cases[0]['entry_ms'],900000)
        self.assertNotEqual(cases[0]['case_id'],others[0]['case_id'])

    def test_IC01_missing_trade_binding_rejects(self):
        task,d=example(); d['trades']=[]
        with self.assertRaisesRegex(ValueError,'TRADE_BINDING'): bind_cases(task,d)

    def test_IC01_duplicate_trade_binding_rejects(self):
        task,d=example(); d['trades']*=2
        with self.assertRaisesRegex(ValueError,'TRADE_BINDING'): bind_cases(task,d)

    def test_IC01_changed_level_count_or_convention_rejects(self):
        for key,value in (('stop',99.),('chosen_reason','SL')):
            task,d=example(); d['ambiguities'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):bind_cases(task,d)
        task,d=example();d['ambiguous_executions_n']=2
        with self.assertRaises(ValueError):bind_cases(task,d)

    def test_IC01_incomplete_parent_rejects(self):
        task,d=example();d['state']='INCOMPLETE'
        with self.assertRaises(ValueError):bind_cases(task,d)

    def test_IC05_different_minutes_stop_first(self):
        rows=[bar(h='100.1',l='99.4'),bar(60000,h='101',l='99.9')]
        result=first_touch(rows,99.5,100.5)
        self.assertEqual(result['state'],'SL_FIRST');self.assertEqual(result['bucket_origin_ms'],0)

    def test_IC05_different_minutes_target_first(self):
        rows=[bar(h='101',l='99.9'),bar(60000,h='100.1',l='99')]
        self.assertEqual(first_touch(rows,99.5,100.5)['state'],'TP_FIRST')

    def test_IC05_both_in_same_bucket_is_unresolved(self):
        result=first_touch([bar()],99.5,100.5)
        self.assertEqual(result['state'],'ORDER_UNRESOLVED_SAME_BUCKET')
        self.assertEqual(result['bucket_origin_ms'],0)

    def test_IC05_open_precedes_later_high_low(self):
        for o,state in (('99','SL_FIRST'),('101','TP_FIRST'),('99.5','SL_FIRST'),('100.5','TP_FIRST')):
            with self.subTest(o=o):
                result=first_touch([bar(o=o,h='102',l='98')],99.5,100.5)
                self.assertEqual(result['state'],state);self.assertEqual(result['trigger'],'OPEN')

    def test_IC05_seconds_have_same_contract(self):
        rows=[bar(h='100.1',l='99.9',step=1000),bar(1000,o='101',h='102',l='99',step=1000)]
        result=first_touch(rows,99.5,100.5,resolution_ms=1000)
        self.assertEqual(result['state'],'TP_FIRST');self.assertEqual(result['bucket_origin_ms'],1000)

    def test_IC05_invalid_or_unverified_data_cannot_resolve(self):
        self.assertEqual(first_touch([bar()],99.5,100.5,verified=False)['state'],'ORDER_UNRESOLVED_DATA')
        for rows in ([],[bar(h='nan')],[bar(),bar(0)], [bar(),bar(120000)]):
            with self.subTest(rows=rows):self.assertTrue(first_touch(rows,99.5,100.5)['state'].startswith('ORDER_UNRESOLVED'))

    def test_IC05_no_touch_is_explicit(self):
        self.assertEqual(first_touch([bar(h='100.1',l='99.9')],99.5,100.5)['state'],'ORDER_UNRESOLVED_NO_TOUCH')

    def test_IC06_denominators_and_contradictions_not_pnl(self):
        a,d=example();rows=bind_cases(a,d)*2
        rows[0]=dict(rows[0],ordering=dict(state='SL_FIRST'))
        rows[1]=dict(rows[1],ordering=dict(state='ORDER_UNRESOLVED_SAME_BUCKET'))
        summary=census_counts(rows)
        self.assertEqual(summary['cases_n'],2);self.assertEqual(summary['resolved_n'],1)
        self.assertEqual(summary['unresolved_n'],1);self.assertEqual(summary['contradicted_n'],1)
        self.assertNotIn('net_pnl_usdt',summary)


if __name__=='__main__':unittest.main()
