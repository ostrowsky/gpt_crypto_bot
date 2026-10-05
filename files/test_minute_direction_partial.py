"""Standing-book prefix sampling and closed-candle availability contracts."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from minute_direction_partial import sample_states, closed_volume

def write_states(path, stamps, shift=0):
    bids=json.dumps([[100+shift-k*.01,1] for k in range(20)])
    asks=json.dumps([[100.01+shift+k*.01,2] for k in range(20)])
    pq.write_table(pa.table({'E':stamps,'bids':[bids]*len(stamps),'asks':[asks]*len(stamps)}),path)

class PartialTests(unittest.TestCase):
    def test_full_state_sampling_no_backfill_stale_reset(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'book.parquet';write_states(path,[1000,2000,11000,21000,22000,80000,81000,91000])
            t,b,s,a=sample_states([path])
            self.assertEqual(t.tolist(),list(range(10000,100001,10000)))
            self.assertTrue(np.isfinite(b[2]).all());self.assertTrue(np.isnan(b[3:7]).all())
            self.assertEqual(s[3],-1);self.assertNotEqual(s[0],s[7]);self.assertEqual(a['stale_samples'],4)
    def test_future_file_change_cannot_change_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            first=Path(d)/'first.parquet';last=Path(d)/'last.parquet'
            write_states(first,[1000,11000,21000,31000]);write_states(last,[41000,51000,61000])
            t,b,s,_=sample_states([first,last]);prefix=sample_states([first])
            write_states(last,[41000,51000,61000],shift=100)
            t2,b2,s2,_=sample_states([first,last])
            mask=t<=30000
            np.testing.assert_array_equal(b[mask],b2[mask]);np.testing.assert_array_equal(s[mask],s2[mask])
            np.testing.assert_array_equal(b[mask],prefix[1][prefix[0]<=30000])
    def test_late_older_state_quarantined_without_sorting_into_past(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.parquet';b=Path(d)/'b.parquet'
            write_states(a,[1000,21000]);write_states(b,[11000,31000],shift=100)
            t,book,segment,audit=sample_states([a,b])
            self.assertEqual(audit['backward_rows'],1)
            self.assertTrue(np.isnan(book[t==20000]).all())
            self.assertAlmostEqual(book[t==30000,0].item(),100.01)
            self.assertAlmostEqual(book[t==40000,0].item(),200.01)
    def test_candle_volume_available_only_after_close_plus_delay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);buf=io.StringIO();w=csv.writer(buf)
            w.writerow(['open_time','o','h','l','c','v','close','quote','trades','buy','buyquote','ignore'])
            w.writerow([0,100,101,99,100,10,59999,1000,7,6,600,0])
            with zipfile.ZipFile(root/'BTCUSDT-1m-2026-03.zip','w') as z:z.writestr('bars.csv',buf.getvalue())
            t=np.arange(10000,180001,10000);flow,bad=closed_volume(root,'BTCUSDT',t)
            self.assertEqual(flow[t==60000,0].item(),0)
            self.assertEqual(flow[t==70000,0].item(),7)
            self.assertFalse(bad[t==70000].item());self.assertTrue(bad[t==130000].item())
            self.assertEqual(flow[t==70000,4].item(),200)

if __name__=='__main__':unittest.main()
