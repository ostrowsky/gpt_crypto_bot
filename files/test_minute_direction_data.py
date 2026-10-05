"""Future mutation, genuine book sequence integrity and closed-bin timing."""
import unittest
import numpy as np
import minute_direction_data as d

def anchor(at=0,uid=10):
    bid=np.arange(100,90,-1.);ask=np.arange(101,111,dtype=float)
    return at,uid,np.r_[np.ones(10,dtype=bool),np.zeros(10,dtype=bool)],np.r_[bid,ask],np.ones(20)

def update(t,first,last,p=100,q=2):
    return t,first,last,np.array([True]),np.array([p],dtype=float),np.array([q],dtype=float)

class BookTests(unittest.TestCase):
    def test_zero_removal_and_absolute_quantity(self):
        b=d.Book();b.anchor(*anchor());b.update(*update(1000,11,11,q=5))
        self.assertEqual(b.bids[100],5)
        b.update(*update(2000,12,12,q=0));self.assertNotIn(100,b.bids)
    def test_atomic_crossing_then_removal_is_valid(self):
        b=d.Book();b.anchor(*anchor())
        b.update(1000,11,12,np.array([False,False]),np.array([99.,99.]),np.array([1.,0.]))
        self.assertTrue(b.valid)
    def test_missing_sequence_invalid_until_new_standing_anchor(self):
        b=d.Book();b.anchor(*anchor());b.update(*update(1000,12,12))
        self.assertFalse(b.valid);self.assertEqual(b.sample(1000)[1],-1)
        b.update(*update(2000,13,13));self.assertFalse(b.valid)
        b.anchor(*anchor(3000,13));self.assertTrue(b.valid)
    def test_overlap_and_covered_updates(self):
        b=d.Book();b.anchor(*anchor());b.update(*update(1000,9,12,q=3))
        self.assertTrue(b.valid);self.assertEqual(b.bids[100],3)
        b.update(*update(2000,11,12,q=50));self.assertEqual(b.bids[100],3)
    def test_stale_crossed_and_future_book_rejected(self):
        b=d.Book();b.anchor(*anchor());self.assertEqual(b.sample(-1)[1],-1)
        self.assertEqual(b.sample(d.STEP+1)[1],-1)
        b.update(*update(1000,11,11,p=101));self.assertFalse(b.valid)
        b.anchor(*anchor(2000,11));b.update(*update(12001,12,12));self.assertFalse(b.valid)
    def test_prune_and_order(self):
        b=d.Book();b.anchor(*anchor())
        prices=np.arange(1,1100,dtype=float)/20
        b.update(1000,11,11,np.ones(len(prices),dtype=bool),prices,np.ones(len(prices)))
        self.assertEqual(len(b.bids),1000)
        row,_=b.sample(1000);self.assertTrue((np.diff(row[2::4])<0).all())
        self.assertTrue((np.diff(row[0::4])>0).all())
    def test_prefix_and_future_mutation_do_not_change_past_books(self):
        events=[update(t,i+11,i+11,q=i+1) for i,t in enumerate(range(1000,100001,1000))]
        full=d.reconstruct(iter(events),[anchor()])
        prefix=d.reconstruct(iter(events[:55]),[anchor()])
        changed=events[:55]+[update(t,i+11,i+11,q=999) for i,t in enumerate(range(56000,100001,1000),start=55)]
        other=d.reconstruct(iter(changed),[anchor()])
        mask=full[0]<=55000
        np.testing.assert_array_equal(full[0][mask],prefix[0][prefix[0]<=55000])
        np.testing.assert_array_equal(full[1][mask],prefix[1][prefix[0]<=55000])
        np.testing.assert_array_equal(full[1][mask],other[1][mask])
    def test_anchor_never_backfills_earlier_grid(self):
        events=[update(t,i+11,i+11) for i,t in enumerate(range(1000,40001,1000))]
        times,books,segments,_=d.reconstruct(iter(events),[anchor(25000,35)])
        self.assertTrue((segments[times<25000]==-1).all())

if __name__=='__main__':unittest.main()
