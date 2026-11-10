"""Construct observed non-resistant-partner exposure from independent partner scores."""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA


def exposure(links,corresponding=False,absolute=False):
    available=links[links.corresponding] if corresponding else links
    column='independent_early_alpha' if absolute else 'partner_R'
    measurable=available[available.independent_n.ge(3)&available[column].notna()].copy()
    measurable['nonresistant_partner']=(measurable[column].ge(.1) if absolute else measurable[column].le(0))
    result=measurable.groupby('author_id').agg(nonresistant_partner=('nonresistant_partner','any'),partner_n=('partner_id','nunique'))
    result['total_partner_n']=available.groupby('author_id').partner_id.nunique().reindex(result.index)
    return result


def verify():
    links=pd.read_parquet(DATA_DIR / 'collaborators/independent_partner_links.parquet')
    checks=[]
    for role,corresponding in [('All',False),('Corresponding',True)]:
        calculated=exposure(links,corresponding)
        for year in [2025,2026]:
            frame=pd.read_parquet(DATA_DIR/f'collaborators/{role}_collaborators_{year}.parquet').set_index('author_id')
            cols=['nonresistant_partner','partner_n','total_partner_n']
            np.testing.assert_array_equal(calculated.reindex(frame.index)[cols].to_numpy(float),frame[cols].to_numpy(float))
            assert frame.early_R.gt(0).all()
            checks.append(dict(role=role,year=year,authors=len(frame),exposed_authors=int(frame.nonresistant_partner.sum())))
    return checks
