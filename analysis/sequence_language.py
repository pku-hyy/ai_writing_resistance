"""Publication-sequence and language-setting supplementary analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
ENGLISH={'AU','CA','IE','NZ','GB','US'}
OUT=ROOT/'results/sequence_language'
SPLINE='bs(early_R,knots=(0.1,0.3),degree=3,include_intercept=False,lower_bound=0,upper_bound=1)'
AGE='bs(age_years,df=4,degree=3,include_intercept=False)'
NUMERIC=['z_history','z_recent','z_team','z_early_n','z_collaborators','z_pre_alpha']
BACKGROUND=AGE+'+'+'+'.join(NUMERIC)

def prepare_language():
    OUT.mkdir(parents=True,exist_ok=True)
    return pd.read_parquet(DATA_DIR / 'sequence_language/language_full_cohort_inputs.parquet'),pd.read_parquet(DATA_DIR / 'sequence_language/language_model_inputs.parquet')
def collect(fit,term,**metadata):
    lo,hi=fit.conf_int().loc[term]
    return dict(**metadata,estimate=fit.params[term],se=fit.bse[term],lower=lo,upper=hi,
                p=fit.pvalues[term],authors=int(fit.nobs),R2=fit.rsquared)

def complete(fit,**metadata):
    return pd.DataFrame({'term':fit.params.index,'estimate':fit.params.to_numpy(),'se':fit.bse.to_numpy(),
       'lower':fit.conf_int().iloc[:,0].to_numpy(),'upper':fit.conf_int().iloc[:,1].to_numpy(),
       'p':fit.pvalues.to_numpy()}).assign(**metadata)

def language(pilot=False):
    base,frame=prepare_language()
    if pilot:
        # The full data design is already inexpensive and checks sparse field support.
        fit=smf.ols('R_late~non_English+'+SPLINE+'+'+BACKGROUND+'+C(field)',frame).fit(cov_type='HC3')
        assert np.isfinite(fit.params).all()
        (OUT/'language_pilot.json').write_text(json.dumps(collect(fit,'non_English'),indent=2))
        print('Language pilot passed',len(frame),'authors',flush=True);return
    estimates=[];coefficients=[]
    formula='non_English+'+SPLINE+'+'+BACKGROUND+'+C(field)'
    for period in ['2025','2026','late']:
        fit=smf.ols('R_'+period+'~'+formula,frame).fit(cov_type='HC3')
        estimates.append(collect(fit,'non_English',model='Binary settings',period=period,
             English=int(frame.non_English.eq(0).sum()),non_English=int(frame.non_English.eq(1).sum())))
        coefficients.append(complete(fit,model='Binary settings',period=period))
        other=smf.ols('R_'+period+'~country+'+SPLINE+'+'+BACKGROUND+'+C(field)',base).fit(cov_type='HC3')
        estimates.append(collect(other,'country',model='Continuous English exposure',period=period))
        coefficients.append(complete(other,model='Continuous English exposure',period=period))
    pd.DataFrame(estimates).to_csv(OUT/'language_adjusted_differences.csv',index=False)
    joint=smf.ols('R_late~'+formula+'+non_English:('+BACKGROUND+')',frame).fit(cov_type='HC3')
    tests=[];sensitivity=[];names=list(joint.params.index)
    for variable in NUMERIC+['academic_age_spline']:
        terms=([x for x in names if 'non_English:' in x and 'bs(age_years' in x]
               if variable=='academic_age_spline' else ['non_English:'+variable])
        contrast=np.zeros((len(terms),len(names)))
        for j,t in enumerate(terms):contrast[j,names.index(t)]=1
        test=joint.wald_test(contrast,scalar=True)
        row=dict(variable=variable,df=len(terms),wald=float(test.statistic),p=float(test.pvalue),authors=len(frame))
        if len(terms)==1:
            row.update(collect(joint,terms[0]));row['English_slope']=joint.params[variable]
            row['non_English_slope']=joint.params[variable]+joint.params[terms[0]]
        tests.append(row)
        addition=AGE if variable=='academic_age_spline' else variable
        single=smf.ols('R_late~'+formula+'+non_English:('+addition+')',frame).fit(cov_type='HC3')
        if variable=='academic_age_spline':
            n=list(single.params.index);terms=[x for x in n if 'non_English:' in x and 'bs(age_years' in x]
            c=np.zeros((len(terms),len(n)))
            for j,t in enumerate(terms):c[j,n.index(t)]=1
            test=single.wald_test(c,scalar=True)
            sensitivity.append(dict(variable=variable,df=len(terms),p=float(test.pvalue),authors=len(frame)))
        else:sensitivity.append(collect(single,'non_English:'+variable,variable=variable,df=1))
        coefficients.append(complete(single,model='Single interaction: '+variable,period='late'))
    for data,name in [(tests,'language_joint_interactions.csv'),(sensitivity,'language_single_interactions.csv')]:
        table=pd.DataFrame(data);table['q_BH']=multipletests(table.p,method='fdr_bh')[1]
        table.to_csv(OUT/name,index=False)
    coefficients.append(complete(joint,model='Joint interactions',period='late'))
    pd.concat(coefficients,ignore_index=True).to_csv(OUT/'language_complete_coefficients.csv',index=False)
    definition=dict(country_list=sorted(ENGLISH),classification='Focal-author affiliation countries on 2023–2024 publications; all observed countries within one language category for binary comparison.',
        cross_setting='No separate model category. Cross-setting authors retained in the continuous English-exposure sensitivity using the original equal-country, equal-year weights.',
        sample='Frozen current cohort; regression only early R>0; no late-outcome selection.',
        support='Primary binary comparison: fields with >=20 early-resistant authors in each category.',
        predictors='Early R spline, academic-age spline, standardized log1p cumulative/recent output, early papers, collaborators, team size; standardized historical alpha; early modal field.',
        uncertainty='OLS, author equal weight, HC3 95% confidence intervals; BH across seven interaction tests separately for joint and single-interaction families.',
        descriptive_denominator='All authors in each pure language category; later bars follow the same initial resistant cohort and require positive R at each intervening year.',
        primary_contrast='Non-English minus English in R units; continuous sensitivity is the opposite directional English-share 0-to-1 contrast.',
        primary_authors=len(frame),primary_fields=frame.field.nunique(),continuous_authors=len(base))
    (OUT/'language_definitions.json').write_text(json.dumps(definition,indent=2))
    print(pd.DataFrame(estimates).to_string(index=False),flush=True)

def sequence(output=None):
    dest=Path(output or OUT);dest.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(DATA_DIR / 'sequence_language/sequence_primary_authors.parquet')
    papers=pd.read_parquet(DATA_DIR / 'papers.parquet').merge(pd.read_parquet(DATA_DIR / 'scores/paper_predictions.parquet')[['author_id','work_id','R']],on=['author_id','work_id'],validate='one_to_one')
    current=papers[papers.author_id.isin(data.index)].sort_values(['author_id','publication_date','work_id'])
    first=current.groupby('author_id').head(5).groupby('author_id').R.mean().reindex(data.index)
    last=current.groupby('author_id').tail(5).groupby('author_id').R.mean().reindex(data.index)
    np.testing.assert_allclose(first,data.first_R,atol=1e-12);np.testing.assert_allclose(last,data.last_R,atol=1e-12)
    bins=pd.qcut(data.first_R,20,labels=False,duplicates='drop').to_numpy();count=np.bincount(bins)
    random=np.random.default_rng(20261026);boot=[]
    for _ in range(500):
        ids=random.integers(0,len(data),len(data));boot.append(np.bincount(bins[ids],weights=last.to_numpy()[ids],minlength=len(count))/np.bincount(bins[ids],minlength=len(count)))
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    frame=pd.DataFrame(dict(bin=np.arange(len(count))+1,authors=count,first_R=np.bincount(bins,weights=first)/count,last_R=np.bincount(bins,weights=last)/count,lower=lo,upper=hi))
    expected=pd.read_csv(SOURCE_DATA / 'sequence_language/sequence_binned_means.csv')
    np.testing.assert_allclose(frame,expected,atol=1e-12);frame.to_csv(dest/'sequence_binned_means.csv',index=False)
    return {'authors':len(data),'Spearman':float(spearmanr(first,last).statistic),'bins':len(count)}


def next_publication(output=None):
    data=pd.read_parquet(DATA_DIR / 'sequence_language/next_publication_inputs.parquet')
    rows=[]
    for width,part in data.groupby('prior_width'):
        weights=1/part.groupby('author_id').author_id.transform('size')
        fitted=smf.wls('R ~ prior_R + C(field_name) + C(quarter)',data=part,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':part.author_id})
        lo,hi=fitted.conf_int().loc['prior_R']
        rows.append(dict(prior_width=int(width),papers=len(part),authors=part.author_id.nunique(),coefficient=fitted.params.prior_R,standard_error=fitted.bse.prior_R,lower=lo,upper=hi,p=fitted.pvalues.prior_R,R2=fitted.rsquared))
    result=pd.DataFrame(rows);expected=pd.read_csv(SOURCE_DATA / 'sequence_language/next_publication_summary.csv')
    np.testing.assert_allclose(result[['coefficient','standard_error','lower','upper']],expected[['coefficient','standard_error','lower','upper']],atol=1e-10)
    if output is not None:Path(output).mkdir(parents=True,exist_ok=True);result.to_csv(Path(output)/'next_publication_summary.csv',index=False)
    return {'models':len(result),'all_estimates_verified':True}


def verify_language(output=None):
    global OUT
    OUT=Path(output or ROOT/'results/sequence_language');OUT.mkdir(parents=True,exist_ok=True)
    language()
    for name in ['language_adjusted_differences.csv','language_joint_interactions.csv','language_single_interactions.csv','language_complete_coefficients.csv']:
        actual=pd.read_csv(OUT/name);expected=pd.read_csv(SOURCE_DATA / 'sequence_language'/name)
        cols=[x for x in ['estimate','se','lower','upper','p','q_BH'] if x in actual]
        np.testing.assert_allclose(actual[cols],expected[cols],atol=1e-10,rtol=1e-6,equal_nan=True)
    return {'authors':len(pd.read_parquet(DATA_DIR / 'sequence_language/language_model_inputs.parquet')),'models_verified':True}
