"""Validate the supplied final analytical cohort and its predictor definitions."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
ENGLISH={'AU','CA','IE','NZ','GB','US'}
SEED=20261018


def country_share(countries):
    countries={c for c in countries if c is not None and str(c).strip()}
    return sum(c in ENGLISH for c in countries)/len(countries) if countries else np.nan


def validate():
    papers=pd.read_parquet(DATA_DIR / 'papers.parquet')
    assert len(papers)==565602 and papers.author_id.nunique()==71133
    assert papers.work_id.is_unique and papers.notna().all().all()
    counts=papers.groupby('author_id').size()
    assert counts.max()<=30 and papers.groupby('author_id').year.nunique().eq(4).all()
    assert not papers.field_name.isin(['Unknown','Mixed','None','nan']).any()
    assert papers.english_share.between(0,1).all() and papers.academic_age.ge(0).all()
    assert papers.alpha.between(0,1).all() and papers.prior_total.ge(papers.prior_12months).all()
    dates=pd.to_datetime(papers.publication_date)
    assert (dates.dt.to_period('Q').astype(str)==papers.quarter).all()
    assert (((dates.dt.year-2023)*4+(dates.dt.month-1)//3)==papers.quarter_index).all()
    folds=papers.drop_duplicates('author_id').set_index('author_id').fold
    calculated=[int(hashlib.sha256(f'{SEED}|{a}'.encode()).hexdigest()[:12],16)%3 for a in folds.index]
    np.testing.assert_array_equal(folds,calculated)
    countries=pd.read_parquet(DATA_DIR / 'definitions/focal_affiliation_countries.parquet')
    countries['calculated_share']=countries.countries.map(country_share)
    joined=papers.merge(countries[['author_id','work_id','calculated_share']],on=['author_id','work_id'],how='left',validate='one_to_one')
    np.testing.assert_allclose(joined.english_share,joined.calculated_share,atol=1e-14)
    return dict(authors=len(counts),papers=len(papers),folds=folds.value_counts().sort_index().to_dict(),year_counts=papers.year.value_counts().sort_index().to_dict(),all_cohort_checks_passed=True)
