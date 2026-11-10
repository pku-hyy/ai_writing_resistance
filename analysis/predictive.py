"""Nested author-fold prediction, grouped Shapley contributions and full model coefficients."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import patsy
from scipy.linalg import solve
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
NAMES=['Early R only','+ Personal background','+ External environment','+ Both groups']
PENALTIES=[1.,10.,100.,1000.]
def design(data, absolute=False):
    blocks, metadata = [], []

    def block(values, group, names, terms):
        blocks.append(np.asarray(values, dtype=float))
        metadata.extend(dict(group=group, label=label, term=term)
                        for label, term in zip(names, terms))

    early = patsy.dmatrix(('1 + bs(early_alpha, knots=(0.025, 0.06), degree=3, include_intercept=False, lower_bound=0, upper_bound=0.1)' if absolute else '1 + bs(early_R, knots=(0.1, 0.3), degree=3, '
                         'include_intercept=False, lower_bound=0, upper_bound=1)'), data)
    block(early, 'Common baseline', ['Intercept'] + [f'Early R: spline {j}' for j in range(1, 6)],
          list(early.design_info.column_names))
    common_n = early.shape[1]
    age = patsy.dmatrix('bs(age_years, df=4, degree=3)-1', data)
    block(age, 'Personal background', [f'Academic age: spline {j}' for j in range(1, 5)],
          list(age.design_info.column_names))
    for name, label in [('history', 'Cumulative first-author output'),
                        ('recent', 'Recent first-author output'),
                        ('early_n', 'Early publication count')]:
        z = data['z_' + name].to_numpy()
        np.testing.assert_allclose(z, (np.log1p(data[name]) - np.log1p(data[name]).mean()) /
                                   np.log1p(data[name]).std(ddof=1), atol=1e-12)
        block(np.column_stack([z, z*z]), 'Personal background',
              [label + ': z', label + ': z squared'], ['z_' + name, 'z_' + name + '^2'])
    pre = data.pre_alpha.to_numpy()
    block(np.column_stack([pre, pre*pre]), 'Personal background',
          ['Historical mean alpha', 'Historical mean alpha: squared'], ['pre_alpha', 'pre_alpha^2'])
    personal_end = sum(b.shape[1] for b in blocks)
    field = pd.Categorical(data.field)
    fields = list(field.categories)
    field_start = personal_end
    block(np.eye(len(fields))[field.codes], 'External environment',
          ['Field: ' + f for f in fields], ['field[' + f + ']' for f in fields])
    block(data.country.to_numpy()[:, None], 'External environment',
          ['English-speaking affiliation share'], ['country'])
    for name, label in [('collaborators', 'Distinct collaborator count'), ('team', 'Mean team size')]:
        z = data['z_' + name].to_numpy()
        np.testing.assert_allclose(z, (np.log1p(data[name]) - np.log1p(data[name]).mean()) /
                                   np.log1p(data[name]).std(ddof=1), atol=1e-12)
        block(np.column_stack([z, z*z]), 'External environment',
              [label + ': z', label + ': z squared'], ['z_' + name, 'z_' + name + '^2'])
    matrix = np.column_stack(blocks)
    groups = [np.arange(common_n), np.arange(personal_end),
              np.r_[np.arange(common_n), np.arange(personal_end, matrix.shape[1])],
              np.arange(matrix.shape[1])]
    reference = field_start + fields.index('Agricultural and Biological Sciences')
    return matrix, pd.DataFrame(metadata), groups, common_n, list(range(field_start, field_start + len(fields))), reference

def coefficients(gram, rhs, indices, common_n, penalty):
    g = gram[np.ix_(indices, indices)].copy()
    g.flat[::len(indices)+1] += np.where(indices < common_n, 0., penalty)
    return solve(g, rhs[indices], assume_a='pos', check_finite=False)

def report_coefficients(beta, indices, size, fields, reference):
    result = np.full(size, np.nan)
    result[indices] = beta
    if reference in indices:
        # Exact re-expression of the original symmetric ridge field effects.
        # Subtracting the reference effect changes no fitted predictions.
        shift = result[reference]
        result[0] += shift
        result[fields] -= shift
    return result
def scores_and_shapley(response, weights, prediction):
    denominator = weights @ ((response - np.average(response, weights=weights)) ** 2)
    errors = np.sum(weights[:, None] * (response[:, None] - prediction) ** 2, axis=0)
    scores = 1 - errors / denominator
    personal = .5 * ((scores[1] - scores[0]) + (scores[3] - scores[2]))
    environment = .5 * ((scores[2] - scores[0]) + (scores[3] - scores[1]))
    assert np.isclose(personal + environment, scores[3] - scores[0], atol=1e-12)
    return scores, np.array([personal, environment])


def calculate(year, absolute=False, bootstraps=200, output=None, verify=True, complete_coefficients=False):
    folder=DATA_DIR/('absolute_alpha/substantive/prediction' if absolute else 'prediction')/str(year)
    data=pd.read_parquet(folder/'author_model_data.parquet')
    bg=pd.read_parquet(DATA_DIR / 'author_background.parquet')
    folds=bg.fold.reindex(data.index).to_numpy(int)
    x,meta,groups,common,fields,reference=design(data,absolute)
    y=data.target_R.to_numpy();n=len(y)
    assert data.notna().all().all() and data.index.is_unique
    prediction=np.full((n,4),np.nan);penalties={};tuning=[]
    for fold in range(3):
        train,test=folds!=fold,folds==fold
        gram=x[train].T@x[train];rhs=x[train].T@y[train]
        inner={f:(x[train&(folds!=f)].T@x[train&(folds!=f)],x[train&(folds!=f)].T@y[train&(folds!=f)]) for f in sorted(set(folds[train]))}
        for m,indices in enumerate(groups):
            choices=[]
            for penalty in ([1.] if m==0 else PENALTIES):
                error=0.
                for f,(g,h) in inner.items():
                    b=coefficients(g,h,indices,common,penalty)
                    error+=np.sum((y[folds==f]-x[folds==f][:,indices]@b)**2)
                choices.append((error,penalty));tuning.append(dict(fold=fold,model=m,penalty=penalty,validation_sse=error))
            penalties[fold,m]=min(choices)[1]
            b=coefficients(gram,rhs,indices,common,penalties[fold,m])
            prediction[test,m]=x[test][:,indices]@b
    scores,contributions=scores_and_shapley(y,np.ones(n),prediction)
    max_difference=float(np.max(np.abs(prediction-np.load(folder/'out_of_fold_predictions.npy'))))
    if verify:np.testing.assert_allclose(prediction,np.load(folder/'out_of_fold_predictions.npy'),atol=2e-10,rtol=1e-9)
    bootstrap=np.empty((bootstraps,6));random=np.random.default_rng(20261021)
    for repeat in range(bootstraps):
        weights=random.multinomial(n,np.full(n,1/n)).astype(float);p=np.full((n,4),np.nan)
        for fold in range(3):
            train,test=folds!=fold,folds==fold
            gram=x[train].T@(weights[train,None]*x[train]);rhs=x[train].T@(weights[train]*y[train])
            for m,indices in enumerate(groups):
                b=coefficients(gram,rhs,indices,common,penalties[fold,m]);p[test,m]=x[test][:,indices]@b
        a,b=scores_and_shapley(y,weights,p);bootstrap[repeat]=np.r_[a,b]
    if verify and bootstraps:
        np.testing.assert_allclose(bootstrap,np.load(folder/'bootstrap_scores.npy')[:bootstraps],rtol=2e-8,atol=2e-10)
    lo,hi=np.quantile(bootstrap,[.025,.975],axis=0) if bootstraps else (np.full(6,np.nan),np.full(6,np.nan))
    model_table=pd.DataFrame(dict(model=NAMES,out_of_sample_R2=scores,lower=lo[:4],upper=hi[:4]))
    group_table=pd.DataFrame(dict(group=['Individual characteristics','Environmental characteristics'],incremental_R2=contributions,lower=lo[4:],upper=hi[4:]))
    report=dict(year=year,absolute=absolute,authors=n,maximum_prediction_difference=max_difference,bootstrap_samples=bootstraps,R2=scores.tolist(),Shapley=contributions.tolist())
    if output is not None:
        dest=Path(output)/('absolute' if absolute else 'main')/str(year);dest.mkdir(parents=True,exist_ok=True)
        model_table.to_csv(dest/'model_scores.csv',index=False);group_table.to_csv(dest/'group_contributions.csv',index=False)
        pd.DataFrame(tuning).to_csv(dest/'penalty_selection.csv',index=False)
        np.save(dest/'out_of_fold_predictions.npy',prediction);np.save(dest/'bootstrap_scores.npy',bootstrap)
        meta.to_csv(dest/'design_columns.csv',index=False)
        (dest/'verification.json').write_text(json.dumps(report,indent=2))
        if complete_coefficients and not absolute:full_coefficients(x,y,folds,groups,common,fields,reference,meta,year,dest,bootstraps)
    print('Prediction verified:',year,'absolute' if absolute else 'main',flush=True)
    return report


def full_coefficients(x,y,folds,groups,common,fields,reference,meta,year,dest,bootstraps=200):
    selected=[];n=len(y)
    for m,indices in enumerate(groups):
        choices=[]
        for penalty in ([0.] if m==0 else PENALTIES):
            error=0.
            for f in range(3):
                train,test=folds!=f,folds==f
                b=coefficients(x[train].T@x[train],x[train].T@y[train],indices,common,penalty)
                error+=np.sum((y[test]-x[test][:,indices]@b)**2)
            choices.append((error,penalty))
        selected.append(min(choices)[1])
    points=[]
    for m,indices in enumerate(groups):
        points.append(report_coefficients(coefficients(x.T@x,x.T@y,indices,common,selected[m]),indices,x.shape[1],fields,reference))
    random=np.random.default_rng(20261026);boot=np.full((bootstraps,4,x.shape[1]),np.nan)
    for repeat in range(bootstraps):
        w=random.multinomial(n,np.full(n,1/n)).astype(float);g=x.T@(w[:,None]*x);h=x.T@(w*y)
        for m,indices in enumerate(groups):boot[repeat,m]=report_coefficients(coefficients(g,h,indices,common,selected[m]),indices,x.shape[1],fields,reference)
    rows=[]
    for m,indices in enumerate(groups):
        for j in indices:
            lo,hi=np.quantile(boot[:,m,j],[.025,.975])
            rows.append(dict(year=year,model=['Baseline','Baseline + personal','Baseline + environment','Full model'][m],term=meta.iloc[j].term,label=meta.iloc[j].label,group=meta.iloc[j].group,estimate=points[m][j],bootstrap_SE=np.std(boot[:,m,j],ddof=1),lower=lo,upper=hi,reference=j==reference,authors=n))
    result=pd.DataFrame(rows);result.to_csv(dest/'complete_coefficients.csv',index=False)
    expected=pd.read_csv(SOURCE_DATA / 'predictive_coefficients/complete_coefficients.csv')
    expected=expected[expected.year.eq(year)]
    np.testing.assert_allclose(result.estimate,expected.estimate,atol=1e-9)
    if bootstraps==200:np.testing.assert_allclose(result[['bootstrap_SE','lower','upper']],expected[['bootstrap_SE','lower','upper']],atol=1e-9)
