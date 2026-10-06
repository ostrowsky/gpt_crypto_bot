import unittest
import json,tempfile
from pathlib import Path
import numpy as np
from verify_order_flow_execution import scalar_price,audit_metrics,verify_spot
from minute_direction_data import sha
from run_order_flow_execution import summarize,METHODS
from test_order_flow_execution import book


class VerificationTest(unittest.TestCase):
    def test_capture_verification_and_clock_regression(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);msg=dict(stream='btcusdt@depth20@100ms',data=dict(lastUpdateId=1,
                bids=[[str(100-i),'1'] for i in range(20)],asks=[[str(101+i),'1'] for i in range(20)]))
            row=dict(receive_monotonic_ns=10,raw=json.dumps(msg));path=d/'messages.jsonl'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            (d/'result.json').write_text(json.dumps(dict(messages_sha256=sha(path),counts={'BTCUSDT:depth':1},invalid=0)))
            self.assertEqual(verify_spot(d)['messages'],1)
            path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,receive_monotonic_ns=9))+'\n',encoding='utf-8')
            (d/'result.json').write_text(json.dumps(dict(messages_sha256=sha(path),counts={'BTCUSDT:depth':2},invalid=0)))
            with self.assertRaises(ValueError):verify_spot(d)

    def test_scalar_depth_and_gap(self):
        a=dict(time=np.arange(8)*1000,book=np.tile(book(),(8,1)),segment=np.zeros(8,int),age=np.zeros(8))
        self.assertAlmostEqual(scalar_price(a,0,3,1,6),(101*2+102)/3*1.00075)
        self.assertIsNone(scalar_price(a,0,11,1,6));a['segment'][4]=-1
        self.assertIsNone(scalar_price(a,0,3,1,6))

    def test_all_summary_metrics_and_corruption(self):
        time=np.tile(np.array([0,86400000,2*86400000,3*86400000,4*86400000]),3)
        symbols=np.repeat(['BTCUSDT','ETHUSDT','SOLUSDT'],5);n=len(time)
        a=dict(time=time,symbol=symbols,mid=np.full(n,100.),immediate=np.tile([101.,99.],(n,1)),
            future=np.tile(np.array([100.,100.])[None,:,None],(n,1,3)))
        for name in METHODS:a['selected_'+name]=np.full((n,2,3),name!='Immediate',bool)
        result=summarize(a);self.assertEqual(audit_metrics(a,result),90)
        result['5']['methods']['OFI']['matched']+=1
        with self.assertRaises(ValueError):audit_metrics(a,result)


if __name__=='__main__':unittest.main()
