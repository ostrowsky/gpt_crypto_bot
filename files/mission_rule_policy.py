"""Same rule-only baseline overrides, without retaining millions of mock calls."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import replay_backtest as rb
import config

def no_model(*args,**kwargs):return None
def no_components(*args,**kwargs):return {}
def zero_bonus(*args,**kwargs):return 0.
def no_events(*args,**kwargs):return {},{}

@contextmanager
def rules_only():
    with ExitStack() as stack:
        for n,f in (('_ml_general_score_replay',no_model),('_ml_trend_nonbull_score_replay',no_model),('_ml_candidate_ranker_components',no_components),('_ml_candidate_ranker_runtime_bonus',zero_bonus),('_load_temporal_scout_events',no_events)):
            stack.enter_context(patch.object(rb,n,new=f))
        stack.enter_context(patch.object(config,'PORTFOLIO_REPLACE_RANKER_ENABLED',False,create=True))
        yield
