"""Evidence tables and common-period plots for the frozen minute benchmark."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=('Prior','Momentum','Logistic','CatBoost','DeepLOB')

def tables(result):
    rows=[];assets=[];paired=[]
    for horizon,models in result['metrics'].items():
        for name,score in models.items():
            scopes={'ALL':score['pooled'],**score['assets']}
            for symbol,s in scopes.items():
                account=result['accounts'][horizon][name]
                row=dict(horizon_min=int(horizon),method=name,asset=symbol,n=s['n'],log_loss=s['log_loss'],
                    brier=s['brier'],macro_f1=s['macro_f1'],balanced_accuracy_pct=100*s['balanced_accuracy'],
                    three_class_accuracy_pct=100*s['accuracy'],majority_class_pct=100*s['base_majority_accuracy'],
                    raw_direction_pct=100*s['raw_direction_correct']/s['raw_direction_n'],
                    raw_majority_pct=100*s['raw_direction_majority_correct']/s['raw_direction_n'],
                    raw_correct=s['raw_direction_correct'],raw_n=s['raw_direction_n'],
                    nonneutral_direction_pct=100*s['nonneutral_correct']/s['nonneutral_n'],
                    nonneutral_majority_pct=100*s['nonneutral_majority_correct']/s['nonneutral_n'],
                    nonneutral_correct=s['nonneutral_correct'],nonneutral_n=s['nonneutral_n'],ece=s['ece'],
                    down_n=s['class_counts'][0],neutral_n=s['class_counts'][1],up_n=s['class_counts'][2])
                if symbol=='ALL':
                    row.update(net_return_pct=account['return_pct'],double_cost_return_pct=account['double_cost_return_pct'],
                        trades=account['trades'],mean_exposure_pct=account['mean_exposure_pct'],
                        max_drawdown_pct=account['max_drawdown_pct'],drawdown_fully_observed=account['drawdown_fully_observed'],
                        missing_holding_marks=account['missing_holding_marks'],funding_initial_capital=account['funding_paid_initial_capital'])
                    rows.append(row)
                else:assets.append(row)
        for comparison,s in result['paired_losses'][horizon].items():
            interval=s.get('familywise95',[None,None])
            paired.append(dict(horizon_min=int(horizon),comparison=comparison,complete_days=s['n_days'],
                mean_logloss_gain=s['mean_loss_gain'],ci_low=interval[0],ci_high=interval[1],robust_claim_allowed=s['robust_claim_allowed']))
    return pd.DataFrame(rows),pd.DataFrame(assets),pd.DataFrame(paired)

def utc(stamp):return pd.to_datetime(stamp,unit='ms',utc=True).isoformat()

def daily_coverage(books,expected):
    from minute_direction_data import sha
    records=[]
    for symbol,meta in expected['assets'].items():
        path=books/(symbol+'.npz')
        if sha(path)!=meta['sha256']:raise ValueError('Coverage input drift')
        with np.load(path,allow_pickle=False) as f:
            time=f['time'];valid=f['segment']>=0
        frame=pd.DataFrame({'day':time//86400000,'valid':valid}).groupby('day').valid.agg(['size','sum'])
        for day,v in frame.iterrows():
            records.append(dict(asset=symbol,date=utc(int(day)*86400000)[:10],clock_samples=int(v['size']),
                valid_states=int(v['sum']),missing_states=int(v['size']-v['sum']),valid_pct=100*v['sum']/v['size']))
    return pd.DataFrame(records)

def render(folder,books=None):
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if result['status']!='COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC':raise ValueError('Incomplete experiment')
    pooled,assets,paired=tables(result)
    pooled.to_csv(folder/'metrics_pooled.csv',index=False);assets.to_csv(folder/'metrics_assets.csv',index=False)
    paired.to_csv(folder/'paired_losses.csv',index=False)
    cuts=result['cuts'];coverage=result['coverage'];counts=result['cohort_counts']
    context=(f"Test: {utc(cuts['test'])} — {utc(cuts['end'])}. "
             f"Train / validation / calibration / test: {counts['train']} / {counts['validation']} / "
             f"{counts['calibration']} / {counts['test']} rows. Inference {counts['inference']} rows; "
             f"unscored future-gap rows {counts['inference']-counts['test']} remain in the execution diagnostic.")
    summary=[]
    for h in (1,3,5):
        block=pooled[pooled.horizon_min==h];winner=block.sort_values('log_loss').iloc[0]
        trained=block[block.method.isin(['CatBoost','DeepLOB'])].sort_values('log_loss').iloc[0]
        summary.append(f"{h} min: lowest log loss {winner.method} ({winner.log_loss:.6f}); "
            f"CatBoost/DeepLOB winner {trained.method}; raw sign accuracy {trained.raw_direction_pct:.2f}% "
            f"({int(trained.raw_correct)}/{int(trained.raw_n)}), observed majority {trained.raw_majority_pct:.2f}%. "
            f"Long-only net {trained.net_return_pct:+.3f}%, trades {int(trained.trades)}.")
    visible=['horizon_min','method','n','log_loss','raw_direction_pct','raw_majority_pct','nonneutral_direction_pct',
             'nonneutral_majority_pct','net_return_pct','trades','mean_exposure_pct','double_cost_return_pct']
    table=pooled[visible].to_markdown(index=False,floatfmt='.4f')
    assettable=assets[['horizon_min','asset','method','n','log_loss','raw_direction_pct','raw_majority_pct',
                       'nonneutral_direction_pct','nonneutral_majority_pct']].to_markdown(index=False,floatfmt='.4f')
    covtable=pd.DataFrame([dict(asset=s,start=utc(v['start']),end=utc(v['end']),samples=v['samples'],valid=v['valid'],
        invalid=v['invalid'],trade_gap_bins=v['trade_gap_bins'],clock_gaps=v['state_audit']['clock_gaps'],
        quarantined_older_states=v['state_audit']['backward_rows']) for s,v in coverage['assets'].items()])
    days=daily_coverage(books,coverage) if books is not None else pd.DataFrame()
    if len(days):days.to_csv(folder/'coverage_daily.csv',index=False)
    notes=[
        'Forecasts classify DOWN / NEUTRAL / UP at exact 1/3/5-minute endpoints; neutral = +/-2bp, not cost break-even.',
        'Raw sign accuracy uses P(up)>P(down), with ties counted DOWN; exact zero realized returns excluded. Nonneutral scores exclude the +/-2bp zone. All denominators and confusion counts are saved.',
        'Majority columns are observed TEST prevalences for comparison, not deployable TEST-fitted predictors. Prior/Momentum/Logistic were fitted on TRAIN only.',
        'Classification tables use identical mature eligible rows for all methods; future-gap exclusion is not used to select trades.',
        'Compact DeepLOB adaptation: CNN + Inception + LSTM, 100 past ten-second states, three heads; not a reproduction of the paper tick-frequency benchmark.',
        'Long-only quote diagnostic: next-ten-second ask entry / bid exit, 7.5bp fee +5bp adverse slippage on each side, actual funding, no leverage, max three positions; signal P(up)>=.5 fixed before testing.',
        'Zero trades means cash and is not a profitable directional strategy. Missing marks mean intragap drawdown is unobserved; execution is idealized and lacks queue/fill parity.',
        'Capital is normalized to one initial-equity unit; order quantity granularity, minimum notionals, queue position and market impact are not simulated.',
        'Late states with clocks below the running maximum are quarantined in recorded order; none are sorted back into the past. Exclusion counts are published.',
        'All 90 available BTC/ETH/SOL March/April files in the pinned dataset used. This is perpetual L2 research, not an October live or actual spot-bot portfolio result.',
        'First June/July incremental archive was rejected after full reconstruction: valid BTC=68, ETH=68, SOL=37 of 500773 samples each, no complete sequence. Missing updates were not repaired with invented states.',
        'Block intervals resample complete UTC days in three-day blocks, with multiplicity across nine comparisons. Fewer than 30 complete days cannot establish registered robust superiority.',
        'No production adoption. Full Truth Harness FAIL TH-11: stale/missing canonical replay source-hash binding; staged change check is separate.'
    ]
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=2,cols=3,subplot_titles=[f'{h} min: log loss (lower is better)' for h in (1,3,5)]+
        [f'{h} min: raw sign accuracy (%)' for h in (1,3,5)])
    for col,h in enumerate((1,3,5),1):
        block=pooled[pooled.horizon_min==h].set_index('method').loc[list(METHODS)]
        fig.add_trace(go.Bar(x=list(METHODS),y=block.log_loss,name=f'{h}m log loss',showlegend=False),row=1,col=col)
        fig.add_trace(go.Bar(x=list(METHODS),y=block.raw_direction_pct,name=f'{h}m sign',showlegend=False),row=2,col=col)
        fig.add_hline(y=float(block.raw_majority_pct.iloc[0]),line_dash='dash',row=2,col=col)
        fig.update_yaxes(range=[0,100],row=2,col=col)
    fig.update_layout(height=650,title='Frozen common TEST cohort: all three assets',template='plotly_white')
    curves=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=[f'{h} min: net capital on the same test clock' for h in (1,3,5)])
    for row,h in enumerate((1,3,5),1):
        for name in METHODS:
            with np.load(folder/f'curve_{name}_{h}.npz') as f:curve=f['curve']
            curves.add_trace(go.Scatter(x=pd.to_datetime(curve[::6,0],unit='ms',utc=True),y=curve[::6,1],name=name,
                legendgroup=name,showlegend=row==1),row=row,col=1)
    curves.update_layout(height=800,template='plotly_white',title='After-cost long-only diagnostic; zero trades = cash')
    # Daily sign scores expose period dependence instead of one interpolated price path.
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as f:pred={k:f[k] for k in f.files}
    daily=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=[f'{h} min: daily raw direction (%)' for h in (1,3,5)])
    for row,j in enumerate(range(3),1):
        mask=pred['scored']&(pred['returns'][:,j]!=0);sample_days=pred['time'][mask]//86400000
        actual=pred['returns'][mask,j]>0
        for name in METHODS:
            correct=(pred[name][mask,j,2]>pred[name][mask,j,0])==actual
            frame=pd.DataFrame({'day':sample_days,'correct':correct}).groupby('day').correct.mean()
            daily.add_trace(go.Scatter(x=pd.to_datetime(frame.index*86400000,unit='ms',utc=True),y=100*frame,
                name=name,legendgroup=name,showlegend=row==1),row=row,col=1)
        daily.update_yaxes(range=[0,100],row=row,col=1)
    daily.update_layout(height=750,template='plotly_white',title='Daily stability on the common scored cohort (partial days included descriptively)')
    # Static shareable plot plus an offline interactive report.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot,axes=plt.subplots(2,3,figsize=(16,8))
    for col,h in enumerate((1,3,5)):
        block=pooled[pooled.horizon_min==h].set_index('method').loc[list(METHODS)]
        axes[0,col].bar(list(METHODS),block.log_loss);axes[0,col].set_title(f'{h} min: log loss')
        axes[1,col].bar(list(METHODS),block.raw_direction_pct);axes[1,col].axhline(float(block.raw_majority_pct.iloc[0]),ls='--',color='black',label='TEST majority')
        axes[1,col].set_ylim(0,100);axes[1,col].set_title(f'{h} min: raw sign accuracy (%)');axes[1,col].legend()
        for ax in axes[:,col]:ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    plot.suptitle('CatBoost / compact DeepLOB — frozen BTC/ETH/SOL perpetual TEST');plot.tight_layout()
    plot.savefig(folder/'comparison.png',dpi=140);plt.close(plot)
    body=f'<h1>CatBoost / DeepLOB: 1, 3, 5 minutes</h1><p>{html.escape(context)}</p>'
    if (folder/'price_forecasts.html').exists():
        body+='<p><a href="price_forecasts.html"><strong>История цены, прогноз и факт для каждой модели — интерактивные графики</strong></a></p>'
    body+='<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary)+'</ul>'
    body+=pooled[visible].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body+=fig.to_html(full_html=False,include_plotlyjs=True)+curves.to_html(full_html=False,include_plotlyjs=False)+daily.to_html(full_html=False,include_plotlyjs=False)
    body+='<h2>Per asset, same eligible rows for every model</h2>'+assets.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body+='<h2>Paired daily block intervals</h2>'+paired.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<h2>Coverage</h2>'+covtable.to_html(index=False)+'<h2>Scope and limitations</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in notes)+'</ul>'
    if len(days):body+='<h2>Daily source coverage: missing states are not direction errors</h2>'+days.to_html(index=False,float_format=lambda x:f'{x:.2f}')
    body+='<p>Data: <a href="https://huggingface.co/datasets/predict-quant/binance-future-orderbook">pinned public perpetual L2</a>; <a href="https://github.com/binance/binance-public-data">official Binance klines</a>; <a href="https://arxiv.org/abs/1808.03668">DeepLOB paper</a>.</p>'
    document='<!doctype html><html><head><meta charset="utf-8"><title>Minute direction benchmark</title><style>body{font:15px sans-serif;max-width:1500px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:7px;border:1px solid #ddd}li{margin:8px 0}</style></head><body>'+body+'</body></html>'
    (folder/'comparison.html').write_text(document,encoding='utf-8')
    report='# CatBoost / DeepLOB: 1, 3, 5 minutes\n\n'+context+'\n\n'+'\n'.join('- '+s for s in summary)
    if (folder/'price_forecasts.html').exists():
        report+='\n\n[История цены, прогноз и факт по каждой модели](price_forecasts.html)\n'
    report+='\n\n'+table+'\n\n## Per asset\n\n'+assettable+'\n\n## Paired daily blocks\n\n'+paired.to_markdown(index=False,floatfmt='.6f')
    report+='\n\n## Coverage\n\n'+covtable.to_markdown(index=False)+'\n\n## Interpretation\n\n'+'\n'.join('- '+s for s in notes)+'\n'
    if len(days):report+='\n\n## Daily source coverage\n\n'+days.to_markdown(index=False,floatfmt='.2f')+'\n'
    (folder/'comparison.md').write_text(report,encoding='utf-8')
    print('\n'.join(summary));print(folder/'comparison.html')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--books',type=Path)
    a=parser.parse_args();render(a.folder,a.books)
