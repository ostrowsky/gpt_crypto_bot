import asyncio,unittest,json,tempfile,hashlib
from pathlib import Path
from unittest.mock import AsyncMock,patch,MagicMock
import numpy as np
import data_collector as dc

class FreshnessTests(unittest.TestCase):
    def test_old_delisting_and_future_bars_never_become_current_training_rows(self):
        data=np.zeros(32,dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')]);data['t']=np.arange(32)*900000;data['c']=100
        for now in (100000.,1.):
            with patch.object(dc,'fetch_klines',new=AsyncMock(return_value=data)),patch.object(dc.time,'time',return_value=now),patch.object(dc,'compute_features') as features,patch.object(dc.critic_dataset,'log_candidate') as record:
                self.assertFalse(asyncio.run(dc._process_coin(None,'OLDUSDT','15m',False,0.)))
                features.assert_not_called();record.assert_not_called()

    def test_asof_registry_filters_break_pairs_with_exact_raw_receipt(self):
        raw=json.dumps(dict(symbols=[dict(symbol='ACTIVEUSDT',status='TRADING',quoteAsset='USDT'),dict(symbol='OLDUSDT',status='BREAK',quoteAsset='USDT')])).encode()
        response=MagicMock();response.read=AsyncMock(return_value=raw);response.__aenter__=AsyncMock(return_value=response)
        session=MagicMock();session.get.return_value=response;session.__aenter__=AsyncMock(return_value=session)
        process=AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as td,patch.object(dc,'COLLECTOR_RECEIPT_DIR',Path(td)),patch.object(dc.config,'load_watchlist',return_value=['ACTIVEUSDT','OLDUSDT']),patch.object(dc.config,'TIMEFRAMES',['15m','1h']),patch.object(dc.aiohttp,'ClientSession',return_value=session),patch.object(dc,'_process_coin',new=process),patch.object(dc.critic_dataset,'append_collector_batch',return_value=dict(new_ids=0,existing_ids=0)),patch.object(dc.critic_dataset,'fill_pending_batch'):
            stats=asyncio.run(dc._collect_once({}))
            self.assertEqual((stats['ok'],stats['total'],stats['watchlist_pairs']), (2,2,4))
            self.assertEqual(len(stats['excluded_pairs']),2);self.assertEqual(stats['coverage_state'],'COMPLETE_TRADABLE')
            self.assertTrue(all(c.args[1]=='ACTIVEUSDT' for c in process.call_args_list))
            receipt=stats['universe_receipt'];self.assertEqual(Path(receipt['path']).read_bytes(),raw)
            self.assertEqual(receipt['sha256'],hashlib.sha256(raw).hexdigest())

if __name__=='__main__':unittest.main()
