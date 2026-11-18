"""RFD09: every original validator transition, bounded chunk state chain."""
from copy import deepcopy
import json
import unittest
from binance_demo_bot import receipt_flow_parallel as parallel
from binance_demo_bot.public_flow_dataset import FlowValidator


def fixture():
    v=FlowValidator();rows=[v.begin_session('one',wall_ms=0,mono_ns=0)]
    for i,t in enumerate((100,200,7000,7100)):
        raw=json.dumps(dict(stream='btcusdt@aggTrade',data=dict(
            e='aggTrade',s='BTCUSDT',a=i,f=i,l=i,E=t,T=t,p='100',q='2',m=False)))
        rows.append(v.ingest(raw,wall_ms=t,mono_ns=t*1000000,session_id='one'))
    rows.append(dict(kind='SESSION_END',wall_ms=7200,mono_ns=7200000000,session_id='one'))
    return rows


class ParallelValidationContract(unittest.TestCase):
    def test_RFD09_chunk_boundaries_match_original_full_state(self):
        rows=fixture();shadow=parallel.BoundaryState();serial=FlowValidator()
        for chunk in (rows[:2],rows[2:4],rows[4:]):
            before=shadow.snapshot()
            for record in chunk:
                parsed=json.loads(record['raw'])['data'] if record['kind']=='MARKET' else None
                shadow.advance(record,parsed)
            after=shadow.snapshot()
            result=parallel.validate_chunk(chunk,before,after)
            self.assertEqual(result,len(chunk))
        self.assertEqual(after['gaps_n'],1)
        self.assertEqual(after['counts']['btcusdt@aggTrade'],4)

    def test_RFD09_forged_validation_and_wrong_seed_cannot_pass(self):
        rows=fixture();shadow=parallel.BoundaryState();before=shadow.snapshot()
        for r in rows:shadow.advance(r,json.loads(r['raw'])['data'] if r['kind']=='MARKET' else None)
        after=shadow.snapshot()
        bad=deepcopy(rows);bad[1]['validation']['continuity']='DUPLICATE'
        with self.assertRaises(ValueError):parallel.validate_chunk(bad,before,after)
        after['counts']['btcusdt@aggTrade']=3
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):parallel.validate_chunk(rows,before,after)


if __name__=='__main__':unittest.main()
