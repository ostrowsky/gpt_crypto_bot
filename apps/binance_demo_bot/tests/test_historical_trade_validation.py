"""HT05/06/07: guarded native trade coverage, minute parity and first touch."""
import copy
from decimal import Decimal
import importlib
import unittest

ORIGIN = 900_000
END = ORIGIN+900_000


def fixture(*, first_prices=None, zero_minutes=(), raw_count=2):
    previous = dict(a=10,p='100.00000000',q='1.00000000',f=90,l=90,
                    T=ORIGIN-1,m=True,M=True)
    trades=[]; minutes=[]; aggregate_id=11; raw_id=91; carried='100.00000000'
    for minute in range(15):
        prices = (first_prices if minute==0 and first_prices is not None else ['100.00000000'])
        if minute in zero_minutes:
            prices=[]
        group=[]
        for price in prices:
            row=dict(a=aggregate_id,p=price,q='2.00000000',f=raw_id,l=raw_id+raw_count-1,
                     T=ORIGIN+minute*60_000+1_000,m=bool(len(trades)%2),M=True)
            trades.append(row);group.append(row);aggregate_id+=1;raw_id+=raw_count
        values=[Decimal(t['p']) for t in group]
        volume=sum((Decimal(t['q']) for t in group),Decimal(0))
        quote=sum((Decimal(t['q'])*Decimal(t['p']) for t in group),Decimal(0))
        buy=sum((Decimal(t['q']) for t in group if not t['m']),Decimal(0))
        buy_quote=sum((Decimal(t['q'])*Decimal(t['p']) for t in group if not t['m']),Decimal(0))
        if values:
            o,h,l,c=values[0],max(values),min(values),values[-1];carried=str(c)
        else:
            o=h=l=c=Decimal(carried)
        at=ORIGIN+minute*60_000
        minutes.append([at,*[format(v,'.8f') for v in (o,h,l,c,volume)],at+59_999,
                        format(quote,'.8f'),sum(t['l']-t['f']+1 for t in group),
                        format(buy,'.8f'),format(buy_quote,'.8f'),'0'])
    following=dict(a=aggregate_id,p='100.00000000',q='1.00000000',f=raw_id,l=raw_id,
                   T=END,m=False,M=True)
    original=[ORIGIN,minutes[0][1],str(max(Decimal(r[2]) for r in minutes)),
              str(min(Decimal(r[3]) for r in minutes)),minutes[-1][4],
              str(sum(Decimal(r[5]) for r in minutes)),END-1,
              str(sum(Decimal(r[7]) for r in minutes)),sum(r[8] for r in minutes),
              str(sum(Decimal(r[9]) for r in minutes)),str(sum(Decimal(r[10]) for r in minutes)),'0']
    return original,minutes,trades,previous,following


class HistoricalTradeValidation(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.historical_trade_validation')

    def check(self, data):
        original,minutes,trades,previous,following=data
        return self.api.validate_trade_window(original,minutes,trades,previous=previous,
                    following=following,received_at_ms=END+1)

    def test_HT05_guarded_complete_window_and_raw_count(self):
        data=fixture();before=copy.deepcopy(data);result=self.check(data)
        self.assertEqual(result['state'],'MATCH')
        self.assertEqual(len(result['minute_checks']),15)
        self.assertEqual(result['aggregate_records_n'],15)
        self.assertEqual(result['raw_trades_n'],30)
        self.assertEqual(data,before)

    def test_HT05_missing_or_wrong_temporal_guard_incomplete(self):
        for which, replacement in ((3,None),(4,None),(3,{'T':ORIGIN}),(4,{'T':END-1})):
            data=list(fixture());data[which]=replacement
            self.assertEqual(self.check(data)['state'],'INCOMPLETE')
        for which,timestamp in ((3,ORIGIN),(4,END-1)):
            data=list(fixture());data[which]['T']=timestamp
            self.assertEqual(self.check(data)['state'],'INCOMPLETE')

    def test_HT05_guard_boundary_aggregate_or_raw_gap(self):
        for which,key in ((3,'a'),(4,'a'),(3,'l'),(4,'f')):
            data=list(fixture());data[which][key]+=1
            self.assertEqual(self.check(data)['state'],'INCOMPLETE')

    def test_HT05_internal_duplicate_gap_overlap_and_time_reversal(self):
        for field,value in (('a',11),('a',13),('f',92),('f',94),('T',ORIGIN)):
            data=fixture();data[2][1][field]=value
            self.assertEqual(self.check(data)['state'],'INCOMPLETE')

    def test_HT04_malformed_records_and_wrong_range(self):
        changes=[{'p':'NaN'},{'p':'0'},{'q':'-1'},{'q':'0'},{'p':100.},
                 {'a':True},{'T':END},{'T':ORIGIN-1},{'m':1},{'M':None},
                 {'f':100,'l':99}]
        for change in changes:
            with self.subTest(change=change):
                data=fixture();data[2][0].update(change)
                self.assertEqual(self.check(data)['state'],'INCOMPLETE')
        data=fixture();del data[2][0]['M']
        self.assertEqual(self.check(data)['state'],'INCOMPLETE')

    def test_HT04_empty_is_not_flat_or_complete(self):
        data=list(fixture());data[2]=[]
        self.assertEqual(self.check(data)['state'],'INCOMPLETE')

    def test_HT06_volume_count_price_and_taker_mismatch(self):
        for change in ({'p':'100.00000001'},{'q':'2.00000010'},
                       {'m':True}):
            with self.subTest(change=change):
                data=fixture();data[2][0].update(change)
                self.assertEqual(self.check(data)['state'],'MISMATCH')
        data=fixture();data[2][0]['l']+=1
        self.assertEqual(self.check(data)['state'],'INCOMPLETE') # conflicts next raw range

    def test_HT06_buyer_maker_false_is_taker_buy(self):
        result=self.check(fixture())
        self.assertEqual(result['minute_checks'][0]['aggregate'][9],'2.00000000')
        self.assertEqual(result['minute_checks'][1]['aggregate'][9],'0')

    def test_HT06_prior_minute_reference_must_match_original(self):
        data=fixture();data[1][0][8]+=1
        self.assertEqual(self.check(data)['state'],'MISMATCH')

    def test_HT06_zero_trade_minute_requires_known_flat_carry(self):
        data=fixture(zero_minutes=(0,1,7,14))
        self.assertEqual(self.check(data)['state'],'MATCH')
        data[1][1][1:5]=['99.00000000']*4;data[0][3]='99.00000000'
        self.assertEqual(self.check(data)['state'],'MISMATCH')

    def test_HT06_fixed_quantization_bound(self):
        for difference, expected in ((Decimal('.00000002'),'MATCH'),(Decimal('.00000003'),'MISMATCH')):
            data=fixture(raw_count=3)
            data[1][0][5]=str(Decimal(data[1][0][5])+difference)
            data[0][5]=str(Decimal(data[0][5])+difference)
            result=self.check(data)
            self.assertEqual(result['state'],expected)
            self.assertEqual(Decimal(result['minute_checks'][0]['differences']['base_volume']['bound']),Decimal('.00000002'))

    def test_HT06_fine_wire_format_cannot_widen_tolerance(self):
        data=fixture();data[2][0]['q']='3'
        self.assertEqual(self.check(data)['state'],'MISMATCH')

    def test_HT05_receipt_before_end_incomplete(self):
        original,minutes,trades,previous,following=fixture()
        result=self.api.validate_trade_window(original,minutes,trades,previous=previous,
                    following=following,received_at_ms=END-1)
        self.assertEqual(result['state'],'INCOMPLETE')

    def test_HT07_tied_timestamp_ids_determine_opposite_first_touches(self):
        for prices, expected in ((['100.00000000','99.40000000','100.60000000'],'TRADE_SEQUENCE_SL_FIRST'),
                                 (['100.00000000','100.60000000','99.40000000'],'TRADE_SEQUENCE_TP_FIRST')):
            data=fixture(first_prices=prices);result=self.check(data)
            self.assertEqual(result['state'],'MATCH')
            resolved=self.api.resolve_bracket(result,data[2],entry_price=100.)
            self.assertEqual(resolved['state'],expected)
            self.assertEqual(resolved['trade_id'],data[2][1]['a'])
            self.assertEqual(resolved['raw_id'],data[2][1]['f'])
            self.assertEqual(resolved['at_ms'],data[2][1]['T'])
            self.assertEqual(resolved['target'],str(100.*1.005))

    def test_HT07_incomplete_or_changed_path_never_resolves(self):
        data=fixture(first_prices=['100.00000000','100.60000000','99.40000000'])
        result=self.check(data);altered=copy.deepcopy(data[2]);altered.pop()
        self.assertEqual(self.api.resolve_bracket(result,altered,entry_price=100.)['state'],'ORDER_UNRESOLVED')
        for state in ('INCOMPLETE','MISMATCH'):
            invalid=dict(result,state=state)
            self.assertEqual(self.api.resolve_bracket(invalid,data[2],entry_price=100.)['state'],'ORDER_UNRESOLVED')

    def test_HT07_no_touch_and_invalid_entry_remain_unresolved(self):
        data=fixture();result=self.check(data)
        for entry in (100.,0.,-1.,float('nan'),float('inf'),True):
            self.assertEqual(self.api.resolve_bracket(result,data[2],entry_price=entry)['state'],'ORDER_UNRESOLVED')


if __name__=='__main__':
    unittest.main()
