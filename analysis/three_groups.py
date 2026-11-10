"""Three-trajectory reference comparisons and subsequent non-resistance models."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import patsy
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
INPUT=DATA_DIR / 'three_groups'
OUT=ROOT/'results/three_groups'
METRICS=['age_mean','age_median','old20','variety','balance','diversity','reuse']
BANDS=['age_0_7','age_8_19','age_20_plus']
COMPONENTS=['fixed_library','subsequent_reuse','first_observed']
GROUPS=['Persistent resistance','Delayed adoption','Stable non-resistance']
PERIODS=['Historical','Early','Late']
BASE_FORMULA=('1+C(field)+country+bs(age_years,df=4,degree=3)+bs(pre_alpha,df=4,degree=3)'
              '+log_history+log_reference_budget+log_team+log_recent+log_early_n')
MEANS=[];CONTRASTS=[];COEFFICIENTS=[];INFLUENCES={}
MECHANISMS=[];MECHANISM_COEFFICIENTS=[]


def groups():
    scores=pd.read_parquet(DATA_DIR / 'scores/author_scores.parquet')
    g=pd.Series('Other trajectory',index=scores.index,name='trajectory')
    g.loc[scores.early.gt(0)&scores.late.gt(0)]=GROUPS[0]
    g.loc[scores.early.gt(0)&scores.late.le(0)]=GROUPS[1]
    g.loc[scores.early.le(0)&scores.late.le(0)]=GROUPS[2]
    return scores,g


def prepare_period(frame,metric,prefix,historical=True):
    selected=frame[frame[metric].notna()].copy()
    means=selected.groupby('author_id').agg(value=(metric,'mean'),valid_n=(metric,'size'),linked=('linked','mean'))
    means['all_n']=frame.groupby('author_id').size()
    means=means[means.valid_n.ge(3 if historical else 1)&means.valid_n.div(means.all_n).ge(.5)]
    selected=selected[selected.author_id.isin(means.index)]
    labels=selected.own_field.replace({'Unknown':np.nan,'nan':np.nan,'None':np.nan})
    fields=pd.crosstab(selected.author_id,labels,normalize='index').iloc[:,1:].add_prefix(prefix+'_field_')
    calendar=selected.groupby('author_id')[[f'calendar_{j}' for j in range(12)]].mean().add_prefix(prefix+'_')
    controls=fields.join(calendar,how='outer')
    if metric in ['variety','balance','diversity','cross_field','reuse']:controls[prefix+'_reference_budget']=np.log1p(means.linked)
    return means,controls


def library_comparisons(bg,annual,history,current,static,context):
    metadata=pd.read_parquet(INPUT/'library_metadata.parquet')
    cctx=annual.groupby(['author_id','period'])[context+['reference_budget','prior_output','prior_output_squared','recent_output']].mean()
    flow=[]
    for section,scheme,filename in [('Initial five-year library','initial_five_years','initial_library_components.parquet'),('Pre-2023 library','pre2023','pre2023_library_components.parquet')]:
        size=metadata[metadata.scheme.eq(scheme)].set_index('author_id').library_size
        ids=size.index
        paper=pd.read_parquet(INPUT/filename)
        v=paper.groupby(['author_id','publication_year'])[COMPONENTS].mean().reset_index()
        v['period']=np.where(v.publication_year.le(2024),'Early','Late')
        v=v.groupby(['author_id','period'])[COMPONENTS].mean().join(cctx)
        for period in ['Early','Late']:
            local=v.xs(period,level='period').reindex(ids).join(bg[['trajectory']]).join(static)
            local['log_library_size']=np.log1p(size)
            controls=list(static)+context+['reference_budget','prior_output','prior_output_squared','recent_output','log_library_size']
            for metric in COMPONENTS:compare(local,controls,metric,period,section)
        flow.append(dict(section=section,authors=len(ids),**{g:int(bg.reindex(ids).trajectory.eq(g).sum()) for g in GROUPS}))
    pd.DataFrame(flow).to_csv(OUT/'library_samples.csv',index=False)
def design(frame,controls):
    c=frame[list(dict.fromkeys(controls))].astype(float)
    assert c.notna().all().all()
    c=c.loc[:,c.std().gt(1e-12)]
    c=(c-c.mean())/c.std()
    core=np.column_stack([np.ones(len(frame)),frame.trajectory.eq(GROUPS[1]),frame.trajectory.eq(GROUPS[0])]).astype(float)
    remainder=c.to_numpy()-core@np.linalg.lstsq(core,c,rcond=None)[0]
    _,r,p=qr(remainder,mode='economic',pivoting=True)
    diagonal=np.abs(np.diag(r));rank=int((diagonal>max(diagonal.max(),1)*1e-9).sum())
    kept=np.sort(p[:rank])
    return np.column_stack([core,c.iloc[:,kept]]), ['Intercept','Delayed adoption','Persistent resistance']+list(c.columns[kept])

def compare(local,controls,metric,period,section,multiplier=1.):
    x,names=design(local,controls)
    fit=sm.OLS(local[metric].to_numpy()*multiplier,x).fit(cov_type='HC3')
    beta,cov=fit.params,fit.cov_params()
    inverse=np.linalg.pinv(x.T@x)
    leverage=np.sum((x@inverse)*x,axis=1)
    residual=fit.resid/np.maximum(1-leverage,1e-8)
    for group,vector in [(GROUPS[2],[1,0,0]),(GROUPS[1],[1,1,0]),(GROUPS[0],[1,0,1])]:
        a=np.zeros(x.shape[1]);a[:3]=vector
        mean=float(a@beta);se=np.sqrt(max(float(a@cov@a),0))
        MEANS.append(dict(section=section,metric=metric,period=period,group=group,
             mean=mean,lower=mean-1.96*se,upper=mean+1.96*se,
             authors=int(local.trajectory.eq(group).sum())))
        influence=(x@inverse@a)*residual
        INFLUENCES[(section,metric,period,group)]=pd.Series(influence,index=local.index)
    for name,vector in [('Delayed − persistent',[0,1,-1]),('Delayed − stable non-resistant',[0,1,0]),
                         ('Persistent − stable non-resistant',[0,0,1])]:
        a=np.zeros(x.shape[1]);a[:3]=vector
        value=float(a@beta);se=np.sqrt(max(float(a@cov@a),0))
        CONTRASTS.append(dict(section=section,metric=metric,period=period,contrast=name,
             estimate=value,se=se,lower=value-1.96*se,upper=value+1.96*se,
             p=2*norm.sf(abs(value/se)),authors=len(local),R2=fit.rsquared))
        influence=(x@inverse@a)*residual
        np.testing.assert_allclose(np.sqrt(influence@influence),se,rtol=1e-5,atol=1e-9)
        INFLUENCES[(section,metric,period,name)]=pd.Series(influence,index=local.index)
    for j,name in enumerate(names):
        COEFFICIENTS.append(dict(section=section,metric=metric,period=period,term=name,
            coefficient=beta[j],se=fit.bse[j],p=fit.pvalues[j]))

def references():
    bg=pd.read_parquet(INPUT/'reference_author_background.parquet')
    annual=pd.read_parquet(INPUT/'reference_author_year.parquet')
    history=pd.read_parquet(INPUT/'historical_paper_metrics.parquet')
    current=pd.read_parquet(INPUT/'current_valid_papers.parquet')
    helper=SimpleNamespace(prepare_period=prepare_period)
    static=patsy.dmatrix(BASE_FORMULA,bg,return_type='dataframe').drop(columns='Intercept').add_prefix('background_')
    context=[c for c in annual if c.startswith('current_')]
    values=annual.groupby(['author_id','period'])[METRICS+['reference_budget','prior_output','prior_output_squared','recent_output']+context].mean()
    historical_context={}
    for metric in METRICS:
        mean,hctx=helper.prepare_period(history,metric,'historical',historical=True)
        hctx=hctx.drop(columns=['historical_reference_budget'],errors='ignore')
        select=history[history[metric].notna()]
        old=select.groupby('author_id')[['reference_budget','prior_output','prior_output_squared','recent_output']].mean()
        old[metric]=mean.value
        if metric=='balance':old['variety']=select.groupby('author_id').variety.mean()
        old=old.join(hctx)
        for period in PERIODS:
            local=old.reindex(bg.index) if period=='Historical' else values.xs(period,level='period').reindex(bg.index)
            local=local.join(bg[['trajectory']]).join(static)
            controls=list(static.columns)+(list(hctx) if period=='Historical' else context)
            if metric in ['variety','balance','diversity','reuse']:controls+=['reference_budget']
            if metric=='balance':controls+=['variety']
            if metric=='reuse':controls+=['prior_output','prior_output_squared','recent_output']
            compare(local,controls,metric,period,'Core reference measures',100 if metric in ['old20','reuse'] else 1)
        historical_context[metric]=hctx
    old=history.groupby('author_id')[BANDS].mean().join(historical_context['age_mean'])
    new=current.groupby(['author_id','publication_year'])[BANDS].mean().reset_index()
    new['period']=np.where(new.publication_year.le(2024),'Early','Late')
    new=new.groupby(['author_id','period'])[BANDS].mean().join(values[context])
    for period in PERIODS:
        local=old.reindex(bg.index) if period=='Historical' else new.xs(period,level='period').reindex(bg.index)
        local=local.join(bg[['trajectory']]).join(static)
        controls=list(static)+(list(historical_context['age_mean']) if period=='Historical' else context)
        for metric in BANDS:compare(local,controls,metric,period,'Reference age bands')
    library_comparisons(bg,annual,history,current,static,context)
    means=pd.DataFrame(MEANS);contrasts=pd.DataFrame(CONTRASTS)
    # The >=20-year outcome is already present in the age-band decomposition.
    # Verify equivalence, then report it once instead of duplicating a test.
    old=means[means.metric.eq('old20')].set_index(['period','group'])
    band=means[means.metric.eq('age_20_plus')].set_index(['period','group']).reindex(old.index)
    np.testing.assert_allclose(old[['mean','lower','upper']],band[['mean','lower','upper']],atol=1e-8)
    means=means[means.metric.ne('old20')]
    contrasts=contrasts[contrasts.metric.ne('old20')]
    contrasts['test_family']=contrasts.section.replace({'Initial five-year library':'Fixed libraries','Pre-2023 library':'Fixed libraries'})
    contrasts['q_BH']=contrasts.groupby(['test_family','contrast']).p.transform(lambda x:multipletests(x,method='fdr_bh')[1])
    means.to_csv(OUT/'reference_adjusted_means.csv',index=False)
    contrasts.to_csv(OUT/'reference_group_contrasts.csv',index=False)
    coefficients=pd.DataFrame(COEFFICIENTS)
    coefficients[coefficients.metric.ne('old20')].to_csv(OUT/'reference_complete_coefficients.csv',index=False)
    changes=[]
    for (section,metric),part in contrasts.groupby(['section','metric']):
        for contrast,local in part.groupby('contrast'):
            early=local[local.period.eq('Early')].iloc[0];late=local[local.period.eq('Late')].iloc[0]
            a=INFLUENCES[(section,metric,'Early',contrast)]
            b=INFLUENCES[(section,metric,'Late',contrast)].reindex(a.index)
            difference=late.estimate-early.estimate
            se=np.sqrt(np.sum((b-a)**2))
            changes.append(dict(section=section,metric=metric,contrast=contrast,estimate=difference,
                lower=difference-1.96*se,upper=difference+1.96*se,p=2*norm.sf(abs(difference/se)),authors=len(a)))
    changes=pd.DataFrame(changes)
    changes['test_family']=changes.section.replace({'Initial five-year library':'Fixed libraries','Pre-2023 library':'Fixed libraries'})
    changes['q_BH']=changes.groupby(['test_family','contrast']).p.transform(lambda x:multipletests(x,method='fdr_bh')[1])
    changes.to_csv(OUT/'reference_early_late_change_contrasts.csv',index=False)
    for section in ['Reference age bands','Initial five-year library','Pre-2023 library']:
        part=means[means.section.eq(section)]
        np.testing.assert_allclose(part.groupby(['period','group'])['mean'].sum(),100,atol=1e-6)
    print('Reference models complete:',len(contrasts)//3,'models;',len(contrasts),'contrasts',flush=True)

def marginal(fitted,x,first,second):
    beta,cov=fitted.params,fitted.cov_params()
    p1,p0=expit(first@beta),expit(second@beta)
    grad=(p1*(1-p1))@first/len(x)-(p0*(1-p0))@second/len(x)
    value=np.mean(p1-p0);se=np.sqrt(max(float(grad@cov@grad),0))
    return value,se

def logit_result(frame,formula,term,label,family,continuous=True):
    x=patsy.dmatrix(formula,frame,return_type='dataframe')
    fit=sm.GLM(frame.exit,x,family=sm.families.Binomial()).fit(cov_type='HC1')
    assert fit.converged
    j=x.columns.get_loc(term);beta,se=fit.params[term],fit.bse[term]
    first,second=x.to_numpy().copy(),x.to_numpy().copy()
    if continuous:
        # Average derivative per SD, without shifting observations outside support.
        p=expit(x.to_numpy()@fit.params);w=p*(1-p)
        effect=beta*w.mean()
        gradient=beta*((w*(1-2*p))@x.to_numpy())/len(x)
        gradient[j]+=w.mean()
        error=np.sqrt(max(float(gradient@fit.cov_params()@gradient),0))
    else:
        first[:,j]=1;second[:,j]=0
        effect,error=marginal(fit,x,first,second)
    result=dict(family=family,predictor=label,term=term,authors=len(frame),exits=int(frame.exit.sum()),
        coefficient=beta,se=se,OR=np.exp(beta),OR_lower=np.exp(beta-1.96*se),OR_upper=np.exp(beta+1.96*se),
        AME_pp=effect*100,AME_lower=(effect-1.96*error)*100,AME_upper=(effect+1.96*error)*100,
        p=fit.pvalues[term],formula=formula)
    MECHANISMS.append(result)
    for name in x:
        MECHANISM_COEFFICIENTS.append(dict(family=family,predictor=label,term=name,
            coefficient=fit.params[name],se=fit.bse[name],p=fit.pvalues[name],authors=len(frame)))
    return result

def mechanisms():
    scores,_=groups()
    data=pd.read_parquet(DATA_DIR / 'prediction/2025/author_model_data.parquet')
    data['exit']=scores.late.reindex(data.index).le(0).astype(int)
    data['late_R']=scores.late.reindex(data.index)
    base=('1+bs(early_R,knots=(0.1,0.3),degree=3,include_intercept=False,lower_bound=0,upper_bound=1)'
          '+bs(age_years,df=4,degree=3)+bs(pre_alpha,df=4,degree=3)+C(field)+country'
          '+z_history+z_recent+z_early_n+z_collaborators+z_team')
    for term,label in [('country','English-speaking affiliation share'),('z_history','Cumulative first-author output'),
                       ('z_recent','Recent first-author output'),('z_early_n','Early publication count'),
                       ('z_collaborators','Distinct collaborator count'),('z_team','Mean team size')]:
        logit_result(data,base,term,label,'Early research setting',term!='country')
    age_design=patsy.dmatrix(base,data,return_type='dataframe')
    age_fit=sm.GLM(data.exit,age_design,family=sm.families.Binomial()).fit(cov_type='HC1')
    age20=np.asarray(patsy.build_design_matrices([age_design.design_info],data.assign(age_years=20))[0])
    age5=np.asarray(patsy.build_design_matrices([age_design.design_info],data.assign(age_years=5))[0])
    effect,error=marginal(age_fit,age_design,age20,age5)
    contrast=(age20-age5).mean(axis=0)
    b=float(contrast@age_fit.params);se=np.sqrt(float(contrast@age_fit.cov_params()@contrast))
    MECHANISMS.append(dict(family='Early research setting',predictor='Academic age: 20 vs 5 years',term='age_contrast',
        authors=len(data),exits=int(data.exit.sum()),coefficient=b,se=se,OR=np.exp(b),OR_lower=np.exp(b-1.96*se),
        OR_upper=np.exp(b+1.96*se),AME_pp=effect*100,AME_lower=(effect-1.96*error)*100,
        AME_upper=(effect+1.96*error)*100,p=2*norm.sf(abs(b/se)),formula=base))
    knowledge=pd.read_parquet(INPUT/'historical_transition_inputs.parquet')
    knowledge_base=base+'+log_pre2023_n+log_reference_budget'
    for metric in METRICS:
        term='history_'+metric
        extra='+history_variety' if metric=='balance' else ''
        logit_result(knowledge,knowledge_base+'+'+term+extra,term,metric,'Historical knowledge practice')
    joint=knowledge_base+'+history_age_mean+history_variety+history_reuse'
    for metric in ['age_mean','variety','reuse']:
        logit_result(knowledge,joint,'history_'+metric,metric,'Joint historical model')
    # A continuous-R sensitivity uses exactly the same authors and predictors.
    sensitivity=[]
    for metric in METRICS:
        term='history_'+metric;formula=knowledge_base+'+'+term+('+history_variety' if metric=='balance' else '')
        x=patsy.dmatrix(formula,knowledge,return_type='dataframe')
        fit=sm.OLS(knowledge.late_R,x).fit(cov_type='HC3')
        b,e=fit.params[term],fit.bse[term]
        sensitivity.append(dict(metric=metric,authors=len(knowledge),coefficient=b,lower=b-1.96*e,upper=b+1.96*e,p=fit.pvalues[term]))
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity['q_BH']=multipletests(sensitivity.p,method='fdr_bh')[1]
    sensitivity.to_csv(OUT/'knowledge_continuous_R_sensitivity.csv',index=False)
    for role in ['All','Corresponding']:
        frame=pd.read_parquet(DATA_DIR/f'collaborators/{role}_collaborators_2025.parquet').set_index('author_id')
        frame['exit']=scores.late.reindex(frame.index).le(0).astype(int)
        formula=('1+nonresistant_partner+bs(early_R,knots=(0.1,0.3),degree=3,include_intercept=False,lower_bound=0,upper_bound=1)'
            '+C(field)+country+bs(age_years,df=4,degree=3)+pre_alpha+log_history+log_recent+log_team'
            '+log_own_n+log_total_partner_n+log_partner_n')
        row=logit_result(frame,formula,'nonresistant_partner',role+' collaborators','Early non-resistant collaborators',False)
        row['exposed_authors']=int(frame.nonresistant_partner.sum())
        row['unexposed_authors']=int(frame.nonresistant_partner.eq(0).sum())
    results=pd.DataFrame(MECHANISMS)
    results['q_BH']=results.groupby('family').p.transform(lambda x:multipletests(x,method='fdr_bh')[1])
    results.to_csv(OUT/'transition_mechanisms.csv',index=False)
    pd.DataFrame(MECHANISM_COEFFICIENTS).to_csv(OUT/'mechanism_complete_coefficients.csv',index=False)
    data.to_parquet(OUT/'transition_author_inputs.parquet')
    knowledge.to_parquet(OUT/'historical_transition_inputs.parquet')
    print(results[['family','predictor','authors','AME_pp','AME_lower','AME_upper','q_BH']].to_string(index=False),flush=True)

def run(output=None):
    global OUT
    OUT=Path(output or OUT);OUT.mkdir(parents=True,exist_ok=True)
    MEANS.clear();CONTRASTS.clear();COEFFICIENTS.clear();INFLUENCES.clear();MECHANISMS.clear();MECHANISM_COEFFICIENTS.clear()
    references();mechanisms()
    for name,columns in [('reference_adjusted_means.csv',['mean','lower','upper']),('reference_group_contrasts.csv',['estimate','se','lower','upper']),('transition_mechanisms.csv',['coefficient','se','AME_pp'])]:
        a=pd.read_csv(OUT/name);b=pd.read_csv(SOURCE_DATA / 'trajectories'/name)
        np.testing.assert_allclose(a[columns],b[columns],rtol=1e-6,atol=1e-8)
    return {'reference_models':len(pd.read_csv(OUT/'reference_group_contrasts.csv'))//3,'transition_tests':len(MECHANISMS),'all_estimates_verified':True}
