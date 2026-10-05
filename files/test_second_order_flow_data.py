import json,unittest
import numpy as np
from second_order_flow_data import Sampler,order_flow,parse_side,validate_quotes

def quotes(n):
    bid=np.tile(np.array([[100-k*.01,10.] for k in range(5)]).ravel(),(n,1))
    ask=np.tile(np.array([[100.02+k*.01,10.] for k in range(5)]).ravel(),(n,1))
    return bid,ask

class OFIDataTests(unittest.TestCase):
    def test_arrow_parser_and_bad_rank_count(self):
        import pyarrow as pa
        row=json.dumps([[f'{100-i*.01:.2f}','1.000'] for i in range(20)])
        value=parse_side(pa.array([row,'bad',json.dumps([['100','1']])]))
        np.testing.assert_allclose(value[0],np.array([[100-i*.01,1] for i in range(5)]).ravel())
        self.assertTrue(np.isnan(value[1:]).all())
    def test_ofi_additions_removals_and_price_changes(self):
        previous=np.tile([100.,10.,101.,10.],(1,5));current=previous.copy()
        current[:,1::4]+=2;np.testing.assert_allclose(order_flow(previous,current),2)
        current=previous.copy();current[:,3::4]+=2;np.testing.assert_allclose(order_flow(previous,current),-2)
        current=previous.copy();current[:,0::4]+=1;np.testing.assert_allclose(order_flow(previous,current),10)
        current=previous.copy();current[:,2::4]-=1;np.testing.assert_allclose(order_flow(previous,current),-10)
    def test_stream_batch_boundaries_equal_whole_prefix_without_backfill(self):
        t=np.arange(100,6100,100);bid,ask=quotes(len(t));bid[:,1::2]+=np.arange(len(t))[:,None]
        full=Sampler().push(t,bid,ask)
        sampler=Sampler();parts=[]
        for start,stop in [(0,7),(7,21),(21,len(t))]:
            block=sampler.push(t[start:stop],bid[start:stop],ask[start:stop])
            if block is not None:parts.append(block)
        for key in full:np.testing.assert_allclose(np.concatenate([p[key] for p in parts]),full[key],equal_nan=True)
        self.assertEqual(full['time'].tolist(),[1000,2000,3000,4000,5000])
        self.assertEqual(full['segment'][0],-1)
        np.testing.assert_allclose(full['flow'][1:],10)
    def test_future_mutation_does_not_change_emitted_history(self):
        t=np.arange(100,8100,100);b,a=quotes(len(t));first=Sampler().push(t,b,a)
        b[t>4000,1::2]*=100;other=Sampler().push(t,b,a)
        mask=first['time']<=4000
        for key in first:np.testing.assert_allclose(first[key][mask],other[key][mask],equal_nan=True)
    def test_late_clocks_and_real_gaps_are_quarantined(self):
        t=np.array([100,200,150,300,400,500,600,700,800,900,1000,5000,5100])
        b,a=quotes(len(t));sampler=Sampler();out=sampler.push(t,b,a)
        self.assertEqual(sampler.audit['backward_rows'],1)
        self.assertTrue((out['segment']<0).all());self.assertTrue(np.isnan(out['book']).all())
        self.assertTrue((out['flow']==0).all())
    def test_crossed_nonpositive_or_unordered_quotes_rejected(self):
        b,a=quotes(3);b[0,0]=a[0,0];b[1,1]=0;b[2,2]=101
        self.assertFalse(validate_quotes(b,a).any())

if __name__=='__main__':unittest.main()
