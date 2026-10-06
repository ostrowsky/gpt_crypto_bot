import unittest
from capture_execution_spot import validate_message


class CaptureTest(unittest.TestCase):
    def test_depth_no_invented_exchange_clock(self):
        a=dict(stream='btcusdt@depth20@100ms',data=dict(lastUpdateId=42,bids=[['100','1']],asks=[['101','1']]))
        self.assertEqual(validate_message(a),('BTCUSDT','depth'))
        self.assertNotIn('E',a['data'])
        a['data']['asks']=[['99','1']]
        with self.assertRaises(ValueError):validate_message(a)

    def test_trade_and_unknown_stream(self):
        a=dict(stream='ethusdt@trade',data=dict(s='ETHUSDT',p='2000',q='1',E=1000,T=999,t=1))
        self.assertEqual(validate_message(a),('ETHUSDT','trade'))
        a['stream']='ethusdt@orders'
        with self.assertRaises(ValueError):validate_message(a)


if __name__=='__main__':unittest.main()
