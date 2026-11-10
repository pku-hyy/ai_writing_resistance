"""Cross-fitted threshold model, contemporary calibration and author resistance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
THRESHOLDS=(.10,)


def numeric(frame):
    age=frame.academic_age.to_numpy(float)
    history=np.log1p(frame.prior_total.to_numpy(float))
    recent=np.log1p(frame.prior_12months.to_numpy(float))
    return np.column_stack([frame.quarter_index,age,age**2,np.log1p(frame.team_size),
        history,history**2,recent,recent**2,frame.english_share,frame.pre2023_alpha])


def author_weights(frame):
    weight=1/frame.groupby('author_id').author_id.transform('size').to_numpy(float)
    return weight/weight.sum()


def fit_model(frame,penalty=1e-5):
    encoder=OneHotEncoder(handle_unknown='ignore',dtype=np.float64)
    scaler=StandardScaler()
    design=sparse.hstack([np.ones((len(frame),1)),scaler.fit_transform(numeric(frame)),
        encoder.fit_transform(frame[['field_name']])],format='csr')
    target=frame.binary.to_numpy(float);weight=author_weights(frame)
    initial=np.zeros(design.shape[1]);mean=np.clip(weight@target,1e-8,1-1e-8)
    initial[0]=np.log(mean/(1-mean))
    def objective(beta):
        linear=design@beta
        loss=weight@(np.logaddexp(0,linear)-target*linear)+penalty*np.dot(beta[1:],beta[1:])/2
        gradient=np.asarray(design.T@(weight*(expit(linear)-target))).ravel()
        gradient[1:]+=penalty*beta[1:]
        return loss,gradient
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',
        options={'maxiter':800,'gtol':1e-7,'ftol':1e-11,'maxcor':20})
    assert fit.success,fit.message
    return dict(encoder=encoder,scaler=scaler,beta=fit.x,iterations=int(fit.nit))


def predict(model,frame):
    design=sparse.hstack([np.ones((len(frame),1)),model['scaler'].transform(numeric(frame)),
        model['encoder'].transform(frame[['field_name']])],format='csr')
    return expit(design@model['beta'])


def calibration_groups(frame):
    frame=frame.copy()
    definitions={
        'history_group':('prior_total',[-np.inf,5,15,40,100,np.inf],['0–5','6–15','16–40','41–100','101+']),
        'recent_group':('prior_12months',[-np.inf,0,1,2,4,np.inf],['0','1','2','3–4','5+']),
        'team_group':('team_size',[-np.inf,1,3,6,10,np.inf],['1','2–3','4–6','7–10','11+']),
        'expected_group':('expected_calibrated',[-np.inf,.1,.2,.3,.4,.5,np.inf],['0–0.1','0.1–0.2','0.2–0.3','0.3–0.4','0.4–0.5','0.5–1'])}
    for name,(column,bins,labels) in definitions.items():
        values=pd.cut(frame[column],bins=bins,labels=labels).astype('object')
        assert values.notna().all()
        frame[name]=values.astype(str)
    frame['field_quarter']=frame.field_name+'|'+frame.quarter
    frame['pre2023_alpha_group']=pd.cut(frame.pre2023_alpha,[-np.inf,.05,.1,.2,.4,np.inf]).astype(str)
    return frame


def calibration_numeric(frame):
    age=np.asarray(patsy.dmatrix('bs(age, knots=(5, 15, 30), degree=3, '
        'include_intercept=False, lower_bound=0, upper_bound=100)-1',{'age':frame.academic_age.to_numpy(float)}))
    return sparse.csr_matrix(np.column_stack([age,frame.english_share.to_numpy(float)]))


def estimate(threshold=.10,output=None,verify=True):
    papers=pd.read_parquet(DATA_DIR / 'papers.parquet')
    history=pd.read_parquet(DATA_DIR / 'pre2023_alpha.parquet')
    frame=papers.join(history[['pre2023_alpha']],on='author_id')
    assert frame.notna().all().all()
    frame['binary']=frame.alpha.ge(threshold).astype(float)
    predictions=np.full(len(frame),np.nan);models=[]
    for fold in range(3):
        training=frame[frame.year.le(2024)&frame.fold.ne(fold)]
        target=frame[frame.fold.eq(fold)]
        assert not set(training.author_id)&set(target.author_id)
        model=fit_model(training);models.append(model)
        predictions[target.index]=predict(model,target)
        print(f'R threshold {threshold:.2f}, fold {fold}: {model["iterations"]} iterations',flush=True)
    calibrated=predictions.copy()
    features=['field_quarter','history_group','recent_group','team_group','expected_group','pre2023_alpha_group']
    for fold in range(3):
        for year in [2025,2026]:
            peers=frame[frame.year.eq(year)&frame.fold.ne(fold)].copy()
            target=frame[frame.year.eq(year)&frame.fold.eq(fold)].copy()
            peers['raw_prediction']=predict(models[fold],peers)
            target['raw_prediction']=predictions[target.index]
            peers['error']=peers.binary-peers.raw_prediction
            overall=peers.groupby(['author_id','quarter']).error.mean().groupby('quarter').mean()
            local=peers.groupby(['author_id','quarter','field_name']).error.mean().groupby(['quarter','field_name']).agg(['sum','count'])
            blocks=[]
            for block in [peers,target]:
                cells=pd.MultiIndex.from_frame(block[['quarter','field_name']])
                drift=(local['sum'].reindex(cells).fillna(0).to_numpy()+20*block.quarter.map(overall).to_numpy())/(local['count'].reindex(cells).fillna(0).to_numpy()+20)
                block['expected_calibrated']=(block.raw_prediction+drift).clip(0,1)
                blocks.append(calibration_groups(block))
            peer,target=blocks
            encoder=OneHotEncoder(handle_unknown='ignore',dtype=float)
            design=sparse.hstack([encoder.fit_transform(peer[features]),calibration_numeric(peer)],format='csr')
            weights=1/peer.groupby('author_id').author_id.transform('size').to_numpy()
            weights*=len(weights)/weights.sum()
            regression=Ridge(alpha=100,solver='lsqr',tol=1e-7,max_iter=3000)
            regression.fit(design,peer.binary-peer.expected_calibrated,sample_weight=weights)
            target_design=sparse.hstack([encoder.transform(target[features]),calibration_numeric(target)],format='csr')
            calibrated[target.index]=(target.expected_calibrated+regression.predict(target_design)).clip(0,1)
    result=papers[['author_id','work_id','year','alpha']].copy()
    result['expected_early']=predictions;result['expected_calibrated']=calibrated
    result['R']=calibrated-frame.binary
    annual=result.groupby(['author_id','year']).R.mean().unstack()
    annual.columns=annual.columns.astype(str)
    annual['early']=annual[['2023','2024']].mean(axis=1)
    annual['late']=annual[['2025','2026']].mean(axis=1)
    report=dict(threshold=threshold,authors=len(annual),papers=len(result))
    if verify:
        expected_path=(DATA_DIR / 'scores/author_scores.parquet' if np.isclose(threshold,.1)
            else DATA_DIR/f'scores/thresholds/threshold_{threshold:.2f}_author_scores.parquet')
        expected=pd.read_parquet(expected_path).reindex(annual.index)
        difference=np.max(np.abs(annual[expected.columns].to_numpy()-expected.to_numpy()))
        report['maximum_author_score_difference']=float(difference)
        np.testing.assert_allclose(annual[expected.columns],expected,rtol=1e-5,atol=3e-6)
        if np.isclose(threshold,.1):
            old=pd.read_parquet(DATA_DIR / 'scores/paper_predictions.parquet')
            np.testing.assert_allclose(result.expected_calibrated,old.expected_calibrated,rtol=1e-5,atol=4e-6)
    if output is not None:
        folder=Path(output)/f'threshold_{threshold:.2f}';folder.mkdir(parents=True,exist_ok=True)
        result.to_parquet(folder/'paper_predictions.parquet',index=False)
        annual.to_parquet(folder/'author_scores.parquet')
        (folder/'verification.json').write_text(json.dumps(report,indent=2))
    return result,annual,report
