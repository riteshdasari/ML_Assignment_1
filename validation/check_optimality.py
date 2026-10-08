import os, sys, json, csv, warnings, time

from pathlib import Path
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
start=time.time()
result={'method':'Targeted checks, cached fold transformations, Gram matrices for coordinate descent; not an exhaustive search','elastic':[],'local':[],'seed_check':[]}
(ROOT / 'results').mkdir(exist_ok=True)
def save(): (ROOT / 'results' / 'optimality_check.json').write_text(json.dumps(result,indent=2))
def read(v):
    with open(ROOT / 'data' / f'IMT2024080_train_var{v}(in).csv') as f:
        r=csv.reader(f); next(r); a=np.array(list(r),float)
    return a[:,:-1],a[:,-1]
data={v:read(v) for v in [1,2]}
cache={}
def folds(v,d,seed):
    key=(v,d,seed)
    if key in cache:return cache[key]
    X,y=data[v]; entries=[]
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(X):
        poly=PolynomialFeatures(d,include_bias=False)
        scaler=StandardScaler()
        a=scaler.fit_transform(poly.fit_transform(X[tr])); b=scaler.transform(poly.transform(X[va]))
        # Explicit centering is equivalent to the fitted intercept. Reuse its Gram matrix.
        xm=a.mean(axis=0); ym=y[tr].mean(); a=np.asfortranarray(a-xm); b=b-xm
        entries.append((a,b,y[tr]-ym,y[va],ym,np.ascontiguousarray(a.T@a)))
    cache[key]=entries; return entries
def summarize(scores):
    return {'mse':float(np.mean([s['mse'] for s in scores])), 'r2':float(np.mean([s['r2'] for s in scores])), 'folds':scores}
def fit_scores(v,d,alpha,seed=42,kind='lasso',ratio=None):
    scores=[]
    for a,b,y,truth,ym,g in folds(v,d,seed):
        m=Ridge(alpha=alpha,fit_intercept=False) if kind=='ridge' else Lasso(alpha=alpha,tol=1e-5,max_iter=100000,fit_intercept=False,precompute=g)
        if kind=='elastic':m=ElasticNet(alpha=alpha,l1_ratio=ratio,tol=1e-5,max_iter=200000,fit_intercept=False,precompute=g)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning); m.fit(a,y)
        pred=m.predict(b)+ym
        s={'mse':mean_squared_error(truth,pred),'r2':r2_score(truth,pred),'warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught)}
        if kind!='ridge':s.update(iterations=int(m.n_iter_),dual_gap=float(m.dual_gap_))
        scores.append(s)
    return {'var':v,'degree':d,'alpha':alpha,'model':kind,'ratio':ratio,'seed':seed,**summarize(scores)}
with threadpool_limits(limits=2):
    for ratio in [.25,.5]:
        original=[]; extended=[]
        for i,(a,b,y,truth,ym,g) in enumerate(folds(2,11,42)):
            m=ElasticNet(alpha=.001,l1_ratio=ratio,tol=1e-5,max_iter=30000,fit_intercept=False,precompute=g,warm_start=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning); m.fit(a,y)
            pred=m.predict(b)+ym; first_iter=int(m.n_iter_)
            original.append({'mse':mean_squared_error(truth,pred),'r2':r2_score(truth,pred),'iterations':first_iter,'dual_gap':float(m.dual_gap_),'warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught)})
            m.max_iter=170000
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning); m.fit(a,y)
            pred=m.predict(b)+ym
            extended.append({'mse':mean_squared_error(truth,pred),'r2':r2_score(truth,pred),'iterations_additional':int(m.n_iter_),'iterations_total':first_iter+int(m.n_iter_),'dual_gap':float(m.dual_gap_),'warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught)})
            print(f'Elastic ratio={ratio}, fold={i+1}: MSE {original[-1]["mse"]:.7f} -> {extended[-1]["mse"]:.7f}; total iterations {extended[-1]["iterations_total"]}; warnings {extended[-1]["warnings"]}',flush=True)
        result['elastic'].append({'degree':11,'alpha':.001,'l1_ratio':ratio,'original_30000':summarize(original),'continued_up_to_200000':summarize(extended)}); save()
    for v,ds,alphas,kind in [(1,[4,5,6],[.0075,.00975,.0125],'lasso'),(2,[11,12,13],[1.5,2.5,3.5],'ridge')]:
        for d in ds:
            for alpha in alphas:
                r=fit_scores(v,d,alpha,kind=kind); result['local'].append(r); save()
                print(f'Local var{v} {kind} degree={d} alpha={alpha}: MSE {r["mse"]:.9f}, warnings={sum(f["warnings"] for f in r["folds"])}',flush=True)
        best=min([r for r in result['local'] if r['var']==v],key=lambda r:r['mse'])
        # One extra split assesses sensitivity without repeating the entire search.
        base=(5,.00975) if v==1 else (12,2.5)
        configs=list(dict.fromkeys([base,(best['degree'],best['alpha'])]))
        # The closest original-model challenger supplies a useful ranking check too.
        top=sorted([r for r in result['local'] if r['var']==v],key=lambda r:r['mse'])
        if len(configs)==1:configs.append((top[1]['degree'],top[1]['alpha']))
        for d,alpha in configs:
            r=fit_scores(v,d,alpha,seed=2026,kind=kind); result['seed_check'].append(r); save()
            print(f'Seed2026 var{v} degree={d} alpha={alpha}: MSE {r["mse"]:.9f}',flush=True)
    result['elapsed_seconds']=time.time()-start; save()
print('CHECK COMPLETE',flush=True)
