from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np

from run_capacity_catboost import mission, publish, validated_trace, SOURCES
from historical_signal_evaluation import sha
from capacity_catboost import BAR


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.day = '2026-09-14'
        start = int(datetime(2026,9,14,tzinfo=ZoneInfo('Europe/Budapest')).timestamp()*1000)
        self.data = np.zeros(96,dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')])
        self.data['t'] = start+np.arange(96)*BAR
        self.data['o'] = 100; self.data['c'] = np.linspace(100,110,96)
        self.clock = start+40*BAR
        self.objective = {'eligible_days':[self.day],'label_pairs':{(self.day,'AAA')}}

    def test_mission_is_posthoc_and_exact_cutoff_is_required(self):
        leader,capture,known = mission(self.data,self.clock,'AAA',self.objective)
        self.assertTrue(leader); self.assertTrue(known); self.assertGreater(capture,0)
        altered = self.data.copy(); altered['c'][40:] = 120
        self.assertNotEqual(capture,mission(altered,self.clock,'AAA',self.objective)[1])
        self.assertFalse(mission(self.data[:50],self.clock,'AAA',self.objective)[2])

    def test_nonleaders_and_after_cutoff_are_not_fake_capture(self):
        self.assertEqual(mission(self.data,self.clock,'BBB',self.objective),(False,None,True))
        self.assertEqual(mission(self.data,int(self.data['t'][94]),'AAA',self.objective),(False,None,False))
        self.assertEqual(mission(self.data,self.clock,'AAA',{'eligible_days':[],'label_pairs':set()}),
                         (False,None,False))

    def test_publication_is_immutable_and_nan_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json'
            publish(p,{'state':'REJECTED'})
            self.assertEqual(json.loads(p.read_bytes()),{'state':'REJECTED'})
            with self.assertRaises(FileExistsError): publish(p,{'state':'PASS'})
            with self.assertRaises(ValueError): publish(Path(tmp)/'bad.json',{'metric':np.nan})

    def test_reused_trace_requires_receipt_sources_and_identical_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'archive'; d=Path(tmp)/'trace'; a.mkdir();d.mkdir()
            publish(a/'manifest.json',{'start_ms':1,'end_ms':9,'eligible_symbols':['AAA']})
            publish(d/'registration.json',{'maximum_available_bounds':[1,9],
                'archive_sha256':sha(a/'manifest.json'),
                'sources':{n:sha(Path(__file__).with_name(n)) for n in SOURCES}})
            publish(d/'result.json',{'arms':{}})
            publish(d/'historical_unsigned.json',{'start_ms':1,'end_ms':9,'universe':['AAA']})
            publish(d/'champion.json',{})
            publish(d/'receipt.json',{p.name:sha(p) for p in d.iterdir()})
            validated_trace(a,d)
            (d/'champion.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'receipt mismatch'): validated_trace(a,d)
            (d/'champion.json').write_text('{}')
            (a/'manifest.json').write_text('{"start_ms":2,"end_ms":9,"eligible_symbols":["AAA"]}')
            with self.assertRaisesRegex(ValueError,'maximum archive'): validated_trace(a,d)


if __name__ == '__main__': unittest.main()
