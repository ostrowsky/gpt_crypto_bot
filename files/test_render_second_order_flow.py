import unittest
import numpy as np
from test_second_order_flow_models import source
from minute_direction_data import SYMBOLS
from second_order_flow_models import HORIZONS,targets
from render_second_order_flow import METHODS,select_origins,build_view,html_document,metric_plot

JS_CHECKER=r'''
const fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(process.argv[1],'utf8'),script=html.match(/<script>const payload=([\s\S]*)<\/script>/)[0].slice(8,-9);
const prefix=`const assert=require('node:assert/strict');let draws=0;
const selector=()=>({value:'',add(o){if(!this.value)this.value=o.value;},addEventListener(){}});
const elements={asset:selector(),origin:selector(),interval:{checked:false,addEventListener(){}},focus:{checked:false,addEventListener(){}},facts:{innerHTML:''},metrics:{innerHTML:''},paired:{innerHTML:''},coverage:{textContent:''}};
const document={getElementById:id=>elements[id]},Option=function(text,value){this.text=text;this.value=value;};
const Plotly={react(id,traces,layout){draws++;assert.equal(id,'chart');
const points=traces.filter(t=>t.mode==='lines+markers');assert.equal(points.length,3);
points.forEach(t=>{assert.equal(t.x.length,7);assert.equal(t.y.length,7);assert(t.y.every(Number.isFinite));assert.equal(Date.parse(t.x[0]),Number(elements.origin.value));});
traces.forEach(t=>assert.equal(t.x.length,t.y.length));
assert.deepEqual(layout.xaxis.range,layout.xaxis2.range);assert.deepEqual(layout.xaxis.range,layout.xaxis3.range);
assert.deepEqual(layout.yaxis.range,layout.yaxis2.range);assert.deepEqual(layout.yaxis.range,layout.yaxis3.range);assert(layout.yaxis.range.every(Number.isFinite));
traces.filter(t=>t.name==='История').forEach(t=>t.x.forEach((x,i)=>{if(Date.parse(x)>Number(elements.origin.value))assert.equal(t.y[i],null);}));
traces.filter(t=>t.name==='Факт после выдачи').forEach(t=>t.x.forEach((x,i)=>{if(Date.parse(x)<Number(elements.origin.value))assert.equal(t.y[i],null);}));
}};
`;
const suffix=`for(const s of payload.symbols)for(const t of payload.origins_ms)for(const focus of [false,true])for(const interval of [false,true]){
elements.asset.value=s;elements.origin.value=String(t);elements.focus.checked=focus;elements.interval.checked=interval;draw();
assert(elements.facts.innerHTML.includes('CatBoost_OFI'));assert(elements.facts.innerHTML.includes('Ridge_OFI'));}
assert.equal(pct(null),'—');console.log('PASS synchronized chart draws:',draws);`;
vm.runInNewContext(prefix+script+suffix,{require,console});
'''

class SecondRenderTests(unittest.TestCase):
    def test_metric_figure_handles_zero_error_and_no_direction_denominators(self):
        import tempfile
        from pathlib import Path
        rows=[dict(horizon_sec=h,n=10,direction_n=0,direction_correct=0,observed_majority_correct=0,mae_bp=0,rmse_bp=0) for h in (5,10,30)]
        result=dict(metrics={m:dict(pooled=rows) for m in list(METHODS)+['Microprice','Zero']})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'figure.png';metric_plot(result,p);self.assertGreater(p.stat().st_size,1000)

    def fixture(self):
        d=source();origin=int(d['time'][500]);p=(d['book'][:,0]+d['book'][:,2])/2
        pred=dict(symbol=np.array(SYMBOLS),time=np.full(3,origin),origin_price=np.full(3,p[500]),actual_returns=np.tile(targets(p,d['segment'])[500],(3,1)),scored=np.ones(3,dtype=bool))
        for name in METHODS:pred[name]=np.tile(np.arange(1,7)*1e-5,(3,1))
        return d,pred,{name:np.full(6,2e-5) for name in METHODS},origin

    def test_paths_are_exact_frozen_outputs_on_common_clock(self):
        d,p,w,t=self.fixture();v=build_view(d,p,w,SYMBOLS[0],t)
        self.assertEqual(v['clock_ms'][0],t-60000);self.assertEqual(v['clock_ms'][-1],t+30000)
        self.assertEqual(v['target_ms'],[t]+[t+h*1000 for h in HORIZONS])
        np.testing.assert_allclose(v['forecasts'][METHODS[0]]['price'][1:],p['origin_price'][0]*np.exp(p[METHODS[0]][0]))

    def test_unknown_future_is_retained_and_does_not_change_forecast(self):
        d,p,w,t=self.fixture();first=build_view(d,p,w,SYMBOLS[0],t)
        d['book'][501:]=np.nan;d['segment'][501:]=-1;p['actual_returns'][:]=np.nan;p['scored'][:]=False
        second=build_view(d,p,w,SYMBOLS[0],t)
        self.assertEqual(first['forecasts'],second['forecasts']);self.assertTrue(all(v is None for v in second['price'][61:]))
        self.assertFalse(second['scored'])

    def test_selection_ignores_scored_flags_and_future_labels(self):
        d,p,w,t=self.fixture();cuts=dict(test=t-1,end=t+3600000)
        a=select_origins(p,cuts);p['actual_returns'][:]=np.nan;p['scored'][:]=False
        self.assertEqual(a,select_origins(p,cuts));self.assertEqual(a,[t])
        p['time'][2]+=5000
        with self.assertRaises(ValueError):select_origins(p,cuts)

    def test_html_selector_and_dynamic_axes_contract(self):
        d,p,w,t=self.fixture();v=build_view(d,p,w,SYMBOLS[0],t)
        payload=dict(methods=list(METHODS),symbols=list(SYMBOLS),horizons=list(HORIZONS),origins_ms=[t],views={s:{str(t):dict(v,symbol=s)} for s in SYMBOLS},metrics=[],paired={},coverage={})
        html=html_document(payload)
        self.assertIn('Plotly.react',html);self.assertIn('connectgaps:false',html)
        self.assertIn('focus?v.origin_ms-5000',html);self.assertIn('if(interval)',html)
        self.assertNotIn('PAYLOAD',html);self.assertNotIn('LIBRARY',html)

    def test_javascript_executes_all_selector_interval_and_focus_states(self):
        import shutil,subprocess,tempfile
        from pathlib import Path
        node=shutil.which('node')
        if node is None:self.skipTest('Node.js unavailable')
        d,p,w,t=self.fixture();v=build_view(d,p,w,SYMBOLS[0],t)
        payload=dict(methods=list(METHODS),symbols=list(SYMBOLS),horizons=list(HORIZONS),origins_ms=[t],views={s:{str(t):dict(v,symbol=s)} for s in SYMBOLS},metrics=[],paired={},coverage={})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'view.html';path.write_text(html_document(payload),encoding='utf-8')
            result=subprocess.run([node,'-e',JS_CHECKER,str(path)],capture_output=True,text=True,encoding='utf-8',check=True)
            self.assertIn('PASS synchronized chart draws: 13',result.stdout)


if __name__=='__main__':unittest.main()
