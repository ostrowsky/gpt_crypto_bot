"""Causal OHLC reference and non-promoting learning diagnostics; not actual PnL."""
import math
from .adapter import _sha


def samples(bars):
    result=[]
    for i in range(3,len(bars)):
        row=bars[i];prior=bars[i-1]
        if any(int(bars[j][0])-int(bars[j-1][0])!=900000 for j in range(i-2,i+1)):continue
        c=float(prior[4]);p=float(bars[i-2][4]);p3=float(bars[i-3][4])
        if min(c,p,p3,float(row[1]))<=0:continue
        x=[c/p-1,c/p3-1,(float(prior[2])-float(prior[3]))/c,math.log1p(float(prior[5]))]
        entry=float(row[1])*1.0001;stop=entry*.995;tp=entry*1.005
        if float(row[3])<=stop:exit_price=stop*.9999;reason='SL'
        elif float(row[2])>=tp:exit_price=tp*.9999;reason='TP'
        else:exit_price=float(row[4])*.9999;reason='TIME'
        net=30*(exit_price/entry-1)-.03-.03*exit_price/entry
        result.append(dict(origin_ms=int(row[0]),feature_end_ms=int(prior[6]),label_end_ms=int(row[6]),
            features=x,net_usdt=net,label=int(net>0),reason=reason))
    return result


def reference_replay(bars):
    rows=[];daily={}
    for row in samples(bars):
        if row['origin_ms']%10_800_000:continue
        day=row['origin_ms']//86_400_000
        if daily.get(day,0)<=-1:continue
        rows.append(row);daily[day]=daily.get(day,0)+row['net_usdt']
    return dict(evidence_kind='SIMULATION',alpha_qualified=False,trades_n=len(rows),
        net_usdt=str(sum(r['net_usdt'] for r in rows)),rows=rows,days_N=len(daily),
        gaps_n=sum(int(bars[i][0])-int(bars[i-1][0])!=900000 for i in range(1,len(bars))),
        fee_per_side='0.001',slippage_per_side='0.0001',historical_receipt_coverage='UNKNOWN',
        verdict='NOT_ALPHA_EVIDENCE',first_open_ms=int(bars[0][0]) if bars else None,
        last_close_ms=int(bars[-1][6]) if bars else None)


def fit_diagnostic(bars,train_fraction=.8):
    data=samples(bars);cut=int(len(data)*train_fraction)
    train=data[:max(0,cut-1)];test=data[cut:]
    if len(train)<100 or len(test)<30:return dict(state='NOT_TRAINED',weights=[],scaler={},promotion_allowed=False)
    means=[sum(r['features'][j] for r in train)/len(train) for j in range(4)]
    scales=[max(1e-8,math.sqrt(sum((r['features'][j]-means[j])**2 for r in train)/len(train))) for j in range(4)]
    weights=[0.]*5
    def feature(row):return [1.]+[max(-10,min(10,(row['features'][j]-means[j])/scales[j])) for j in range(4)]
    def predict(x):return 1/(1+math.exp(-max(-30,min(30,sum(w*v for w,v in zip(weights,x))))))
    for _ in range(3):
        for r in train:
            x=feature(r);error=predict(x)-r['label']
            weights=[w-.002*(error*v+.0001*w) for w,v in zip(weights,x)]
    probs=[predict(feature(r)) for r in test]
    prevalence=sum(r['label'] for r in train)/len(train)
    return dict(state='SIMULATION_CANDIDATE_FITTED',weights=weights,scaler=dict(means=means,scales=scales),
        train_n=len(train),test_N=len(test),train_label_end_ms=train[-1]['label_end_ms'],test_origin_ms=test[0]['origin_ms'],
        brier=sum((p-r['label'])**2 for p,r in zip(probs,test))/len(test),
        correct_n=sum((p>=.5)==bool(r['label']) for p,r in zip(probs,test)),
        train_positive_n=sum(r['label'] for r in train),test_positive_n=sum(r['label'] for r in test),
        constant_baseline_brier=sum((prevalence-r['label'])**2 for r in test)/len(test),
        constant_baseline_correct_n=sum((prevalence>=.5)==bool(r['label']) for r in test),
        train_sha256=_sha(train),source_kind='SIMULATION',promotion_allowed=False)


def learning_handoff(rows,*,fit_cutoff_ms):
    valid=[r for r in rows if r.get('state')=='CLOSED' and type(r.get('label_available_at_ms')) is int
           and r['label_available_at_ms']<=fit_cutoff_ms and r.get('features') is not None
           and r.get('source_kind')=='ACTUAL_DEMO' and r.get('private_delivery_matched') is True
           and all(type(r.get(k)) is int for k in ('feature_end_ms','origin_ms','label_end_ms'))
           and r['feature_end_ms']<r['origin_ms']<=r['label_end_ms']<=r['label_available_at_ms']
           and ('feature_available_at_ms' not in r or (type(r['feature_available_at_ms']) is int and r['feature_available_at_ms']<=r['origin_ms']))]
    return dict(state='NOT_TRAINED',reason='INSUFFICIENT_CHRONOLOGICAL_ACTUAL_SAMPLE',
        eligible_n=len(valid),excluded_n=len(rows)-len(valid),rows_N=len(rows),promotion_allowed=False,
        daily_objective_eligible_n=sum(r.get('daily_objective_eligible') is True for r in valid),
        manifest_sha256=_sha(valid),fit_cutoff_ms=fit_cutoff_ms)
