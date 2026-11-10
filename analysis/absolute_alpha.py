"""Annual absolute-alpha summaries for the four-panel robustness figure."""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
THRESHOLD=.10
YEARS=[2023,2024,2025,2026]
def author_data(papers):
    assert not papers.duplicated(['author_id', 'work_id']).any()
    assert papers.alpha.notna().all() and papers.alpha.between(0, 1).all()
    grouped = papers.groupby(['author_id', 'year']).alpha.agg(['mean', 'size'])
    means = grouped['mean'].unstack().reindex(columns=YEARS)
    counts = grouped['size'].unstack().reindex(columns=YEARS)
    assert means.notna().all().all() and counts.ge(1).all().all()
    frame = means.rename(columns=lambda year:f'alpha_{year}').join(counts.rename(columns=lambda year:f'n_{year}'))
    frame['early_alpha'] = (frame.alpha_2023+frame.alpha_2024)/2
    frame['early_n'] = frame.n_2023+frame.n_2024
    frame['early_below'] = frame.early_alpha.lt(THRESHOLD)
    return frame

def calculate(output=None):
    frame=author_data(pd.read_parquet(DATA_DIR/'papers.parquet'))
    states=frame[['early_alpha','alpha_2025','alpha_2026']].lt(.10)
    result=pd.DataFrame({'period':['early','2025','2026'],'cohort_authors':len(frame),'n':states.sum().to_numpy()})
    result['percent']=100*result.n/result.cohort_authors
    expected=pd.read_csv(SOURCE_DATA/'absolute_alpha_states.csv')
    np.testing.assert_allclose(result[['cohort_authors','n','percent']],expected[['cohort_authors','n','percent']],atol=1e-12)
    if output is not None:
        dest=Path(output)/'absolute_alpha';dest.mkdir(parents=True,exist_ok=True)
        result.to_csv(dest/'score_states.csv',index=False)
    return {'authors':len(frame),'annual_low_score_counts':result.n.tolist(),'definition':'Current period mean alpha below 0.10'}
