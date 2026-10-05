import unittest
from datetime import datetime,timedelta,timezone
import numpy as np

from run_turnover_economics import paired_test_interval,validate_data
from turnover_economics import BAR


class RunnerTests(unittest.TestCase):
    def test_corrected_interval_is_deterministic_and_zero_for_identical_accounts(self):
        start=int(datetime(2026,9,1,22,tzinfo=timezone.utc).timestamp()*1000)
        end=start+35*86400000
        curve=[(t,10000.) for t in range(start,end+1,BAR)]
        a=paired_test_interval(curve,curve,start,end,draws=100)
        b=paired_test_interval(curve,curve,start,end,draws=100)
        self.assertEqual(a,b);self.assertEqual(a['corrected_interval'],[0.,0.])
        self.assertEqual(a['days'],35);self.assertAlmostEqual(a['confidence'],1-.05/3)

    def test_array_validation_rejects_gap_and_future_unclosed_end(self):
        cache={}
        for tf,step in [('15m',BAR),('1h',4*BAR)]:
            d=np.zeros(8*BAR//step,dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')])
            d['t']=np.arange(len(d))*step;d['o']=d['c']=100;d['h']=101;d['l']=99;d['v']=1
            cache['A',tf]=(d,{})
        m={'eligible_symbols':['A'],'archive_start_ms':0,'end_ms':8*BAR}
        validate_data(cache,m)
        cache['A','15m'][0]['t'][-1]+=BAR
        with self.assertRaisesRegex(ValueError,'grid'):validate_data(cache,m)


if __name__=='__main__':unittest.main()
