"""Actual numerical runner integration, separate from minimal app environment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import _support
import tempfile
import unittest


class NumericalRunner(unittest.TestCase):
    def test_training_and_cache_hold_strict_origin_cutoff(self):
        import numpy as np
        from binance_demo_bot import rl_direction_runner as api
        from binance_demo_bot.rl_direction_data import STEP
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);path=root/'bars.npy'
            bars=np.tile([100.,100.,100.,100.,1000.,600.,10.],(100,1))
            bars[:,3]=100+np.arange(100)*.001;bars[:,1]=bars[:,3]+.001
            np.save(path,bars)
            manifest=dict(symbols=['AAAUSDT'],entries={'AAAUSDT':dict(path=str(path),first_ms=0,count=100)},
                global_first_ms=0,cutoff_ms=100*STEP)
            x,y,rows,counts=api.make_training_set(manifest,0,80*STEP)
            self.assertEqual(x.shape[1:],(32,5));self.assertEqual(y.shape[1],2)
            self.assertTrue(all(r['label_available_at_ms']<=80*STEP for r in rows))
            self.assertTrue(all(r['feature_end_ms']<r['origin_ms'] for r in rows))
            class Bundle:
                encoder_sha256='a'*64
                def assert_frozen(self):return True
                def predict(self,values):
                    n=len(values)
                    return dict(latent=np.zeros((n,16),np.float32),
                        cat_proba=np.full((n,2),.6),cat_return=np.full((n,2),.01),
                        gru_proba=np.full((n,2),.7),gru_return=np.full((n,2),.02))
            screen=dict(symbols=['AAAUSDT'],times_ms=np.arange(101)*STEP,
                symbol_indices=np.array([-1]*33+[0]+[-1]*67))
            cached=api.cache_predictions(manifest,screen,Bundle(),root/'cache')
            tape=api.MarketTape(manifest,screen,root/'cache')
            self.assertEqual(cached['rows_n'],17)
            self.assertIsNone(tape.prediction('AAAUSDT',32*STEP))
            self.assertEqual(len(tape.vector('AAAUSDT',33*STEP)),20)
            self.assertAlmostEqual(tape.forecast('AAAUSDT',33*STEP,'GRU')[1],.02)
            env=api.make_training_env(tape,33*STEP,50*STEP,11,episode_steps=8)
            observation,_=env.reset();self.assertEqual(observation.shape,(27,))
            rewards=0
            for _ in range(8):
                observation,reward,terminal,truncated,info=env.step(0);rewards+=reward
            self.assertTrue(terminal);self.assertFalse(truncated);self.assertEqual(rewards,0)
            self.assertEqual(env.invalid_episodes_n,0)
            env.close();tape.close()

    def test_invalid_train_transition_raises_instead_of_positive_terminal(self):
        from binance_demo_bot import rl_direction_runner as api
        from binance_demo_bot.rl_direction_core import STEP
        class Tape:
            def candidate(self,at):return 'AAAUSDT'
            def vector(self,*a):return [0.]*20
            def bar(self,*a):return None
        env=api.make_training_env(Tape(),0,16*STEP,11,episode_steps=8,allowed_starts=[0])
        env.reset()
        with self.assertRaisesRegex(ValueError,'INVALID_TRAINING_TRANSITION'):
            env.step(2)
        self.assertEqual(env.invalid_episodes_n,1)

    def test_train_ranges_exclude_known_gaps_without_filtering_test_candidate(self):
        import numpy as np
        from binance_demo_bot import rl_direction_runner as api
        from binance_demo_bot.rl_direction_data import STEP
        class Tape:
            manifest={'entries':{'AAA':{'first_ms':0}}}
            screen={'symbols':['AAA'],'times_ms':np.arange(60)*STEP,'symbol_indices':np.zeros(60,dtype=int)}
            def _files(self,symbol):
                bars=np.tile([100.,100.,100.,100.,1000.,500.,10.],(60,1));bars[30]=np.nan
                return bars,None,None
        starts,counts=api.training_episode_starts(Tape(),0,40*STEP,episode_steps=8)
        self.assertTrue(len(starts)>0)
        for at in starts:
            self.assertFalse(any(i<=30<i+16 for i in range(at//STEP,at//STEP+8)))
        self.assertGreater(counts['excluded_starts_n'],0)
        self.assertEqual(Tape.screen['symbol_indices'][30],0)


if __name__=='__main__':unittest.main()
