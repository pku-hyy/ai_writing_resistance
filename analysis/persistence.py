"""Author-score bins, current-state flows, field retention."""
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA


def binned(early,late):
    frame=pd.DataFrame({'early':early,'late':late});frame['bin']=pd.qcut(frame.early,40,duplicates='drop')
    result=frame.groupby('bin',observed=True).agg(x=('early','mean'),y=('late','mean'),sd=('late','std'),n=('late','size')).reset_index(drop=True)
    result['half_ci']=1.96*result.sd/np.sqrt(result.n)
    return result


def run(output=None):
    dest=Path(output or ROOT/'results/persistence');dest.mkdir(parents=True,exist_ok=True)
    scores=pd.read_parquet(DATA_DIR / 'scores/author_scores.parquet')
    primary=binned(scores.early,scores.late);expected=pd.read_csv(SOURCE_DATA / 'resistance_persistence.csv')
    np.testing.assert_allclose(primary,expected,atol=1e-12);primary.to_csv(dest/'main_binned_scores.csv',index=False)
    absolute=pd.read_parquet(DATA_DIR / 'absolute_alpha/author_scores_and_counts.parquet')
    raw=binned(absolute.early_alpha,absolute[['alpha_2025','alpha_2026']].mean(axis=1))
    np.testing.assert_allclose(raw,pd.read_csv(SOURCE_DATA / 'alpha_persistence/continuous_alpha_early_late_source.csv'),atol=1e-12)
    raw.to_csv(dest/'absolute_binned_scores.csv',index=False)
    for kind,states in [('R',scores[['early','2025','2026']].gt(0)),('alpha',absolute[['early_alpha','alpha_2025','alpha_2026']].lt(.1))]:
        stages=['early','2025','2026'];state_rows=[];links=[]
        for j,stage in enumerate(stages):
            n=int(states.iloc[:,j].sum());state_rows.append(dict(period=stage,authors=len(states),positive=n,percent=100*n/len(states)))
        for j in range(2):
            for a in [0,1]:
                for b in [0,1]:links.append(dict(period_from=stages[j],period_to=stages[j+1],state_from=a,state_to=b,authors=int((states.iloc[:,j].eq(a)&states.iloc[:,j+1].eq(b)).sum())))
        pd.DataFrame(state_rows).to_csv(dest/f'{kind}_states.csv',index=False);pd.DataFrame(links).to_csv(dest/f'{kind}_flows.csv',index=False)
    bg=pd.read_parquet(DATA_DIR / 'author_background.parquet');fields=[]
    for field,local in bg.groupby('field'):
        s=scores.reindex(local.index);early=s.early.gt(0);active=early.copy()
        row=dict(field=field,n=len(s),early=100*early.mean())
        for year in ['2025','2026']:active &= s[year].gt(0);row[year]=100*active.sum()/early.sum()
        fields.append(row)
    field=pd.DataFrame(fields).set_index('field');expected=pd.read_csv(SOURCE_DATA / 'field_prevalence_retention.csv').set_index('field')
    np.testing.assert_allclose(field.loc[expected.index,['n','early','2025','2026']],expected[['n','early','2025','2026']],atol=1e-12)
    field.to_csv(dest/'field_prevalence_retention.csv')
    return dict(authors=len(scores),R_bins=len(primary),alpha_bins=len(raw),fields=len(field),all_results_verified=True)
