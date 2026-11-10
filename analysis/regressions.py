"""Re-fit the main and absolute-score OLS models with HC3 uncertainty."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA


def ols_hc3(x,y):
    inverse=np.linalg.pinv(x.T@x)
    beta=inverse@x.T@y;residual=y-x@beta
    leverage=np.sum((x@inverse)*x,axis=1)
    score=x*(residual/np.maximum(1-leverage,1e-8))[:,None]
    covariance=inverse@(score.T@score)@inverse
    return beta,covariance,leverage


def run(output=None):
    from analysis.collaborators import verify as verify_exposure
    exposure_checks=verify_exposure()
    dest=Path(output or ROOT/'results/regressions');dest.mkdir(parents=True,exist_ok=True)
    rows=[];coefficients=[];means=[]
    entries=json.loads((DATA_DIR / 'regression_inputs/models.json').read_text())
    for entry in entries:
        with np.load(DATA_DIR / 'regression_inputs'/entry['matrix'],allow_pickle=False) as z:x=z['design'];names=z['columns'].astype(str).tolist()
        with np.load(DATA_DIR / 'regression_inputs'/entry['outcome'],allow_pickle=False) as z:y=z['outcome']
        beta,cov,lev=ols_hc3(x,y);j=names.index(entry['term']);se=np.sqrt(max(cov[j,j],0))
        np.testing.assert_allclose([beta[j],se],[entry['expected']['estimate'],entry['expected']['se']],atol=1e-9,rtol=1e-8)
        rows.append(dict(analysis='main',name=entry['name'],family=entry['family'],authors=len(y),estimate=beta[j],se=se,lower=beta[j]-1.96*se,upper=beta[j]+1.96*se,p=2*norm.sf(abs(beta[j]/se))))
        for k,name in enumerate(names):
            e=np.sqrt(max(cov[k,k],0));coefficients.append(dict(analysis='main',model=entry['name'],term=name,estimate=beta[k],se=e,lower=beta[k]-1.96*e,upper=beta[k]+1.96*e,p=2*norm.sf(abs(beta[k]/e)) if e else np.nan))
        if entry['family']=='references':
            for group in [0,1]:
                a=np.zeros(x.shape[1]);a[0]=1;a[j]=group;e=np.sqrt(max(a@cov@a,0));v=a@beta
                means.append(dict(model=entry['name'],resistant=group,mean=v,lower=v-1.96*e,upper=v+1.96*e))
    absolute=DATA_DIR / 'absolute_alpha/substantive'
    expected=pd.read_csv(SOURCE_DATA / 'absolute_alpha/partner_coefficients.csv')
    for p in sorted(absolute.rglob('*_verification.npz')):
        with np.load(p,allow_pickle=False) as z:x=z['design'];y=z['outcome'];names=z['columns'].astype(str).tolist()
        beta,cov,_=ols_hc3(x,y);j=names.index('nonlow_partner');se=np.sqrt(max(cov[j,j],0))
        kind,year=p.stem.replace('_verification','').rsplit('_',1)
        e=expected[expected.kind.eq(kind.replace('_',' '))&expected.year.eq(int(year))].iloc[0]
        np.testing.assert_allclose([beta[j],se],[e.coefficient,e.se],atol=1e-8)
        rows.append(dict(analysis='absolute',name=p.stem,family='collaborators',authors=len(y),estimate=beta[j],se=se,lower=beta[j]-1.96*se,upper=beta[j]+1.96*se,p=2*norm.sf(abs(beta[j]/se))))
    pd.DataFrame(rows).to_csv(dest/'contrasts.csv',index=False)
    pd.DataFrame(coefficients).to_csv(dest/'complete_main_coefficients.csv',index=False)
    pd.DataFrame(means).to_csv(dest/'adjusted_reference_means.csv',index=False)
    return dict(models=len(rows),main_models=len(entries),collaborator_exposure_checks=len(exposure_checks),all_estimates_verified=True)
