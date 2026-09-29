import json
from pathlib import Path
import tempfile
import unittest
from recover_rocket_history import merge_interior, exchange_rows, valid


def row(t,c=100): return dict(t=t,o=100,h=110,l=90,c=c,v=5)


class RecoveryTests(unittest.TestCase):
    def test_terminal_partial_does_not_poison_later_closed(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.json',Path(d)/'b.json'
            a.write_text(json.dumps([row(0),row(900000,101)]))
            b.write_text(json.dumps([row(900000,105),row(1800000)]))
            rows,conflicts,errors,_=merge_interior([a,b],0,2700000,900000)
            self.assertEqual(rows[900000]['c'],105)
            self.assertNotIn(1800000,rows)
            self.assertFalse(conflicts)
            self.assertEqual(errors,0)

    def test_real_interior_conflict_requires_exchange(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i in (0,1):
                p=Path(d)/str(i); p.write_text(json.dumps([row(0,100+i),row(900000)])); paths.append(p)
            rows,conflicts,_,_=merge_interior(paths,0,1800000,900000)
            self.assertNotIn(0,rows)
            self.assertEqual(conflicts,{0})

    def test_exchange_active_and_invalid_close_time_rejected(self):
        raw=[[0,100,110,90,105,5,899999]]
        self.assertEqual(exchange_rows(raw,900000,900000)[0]['c'],105)
        self.assertEqual(exchange_rows(raw,900000,899999),[])
        self.assertFalse(valid(row(1),900000))
        self.assertFalse(valid(row(0,float('nan')),900000))


if __name__=='__main__': unittest.main()
