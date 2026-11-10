"""Publication-level reference measures used before author aggregation.

The submission data include these measured outcomes. These functions describe
and implement their construction from linked, deduplicated reference records.
"""
from collections import Counter
from datetime import date
import numpy as np


def reference_age(citing_year,cited_years):
    cited_years=list(cited_years)
    ages=np.asarray([citing_year-y for y in cited_years if y is not None and 1000<=y<=citing_year],float)
    if len(ages)<10 or len(ages)<.5*len(cited_years):return dict(age_mean=np.nan,age_median=np.nan,age_0_7=np.nan,age_8_19=np.nan,age_20_plus=np.nan)
    return dict(age_mean=float(ages.mean()),age_median=float(np.median(ages)),age_0_7=float(np.mean(ages<=7)),age_8_19=float(np.mean((ages>=8)&(ages<=19))),age_20_plus=float(np.mean(ages>=20)))


def disciplinary_composition(subfields):
    subfields=list(subfields)
    counts=np.array(list(Counter(s for s in subfields if s is not None).values()),float)
    if counts.sum()<10 or counts.sum()<.5*len(subfields):return dict(variety=np.nan,balance=np.nan,diversity=np.nan)
    q=counts/counts.sum();k=len(q)
    return dict(variety=k,balance=float(-np.sum(q*np.log(q))/np.log(k)) if k>1 else np.nan,diversity=float(1-np.sum(q*q)))


def reuse_by_publication(records):
    """Yield (work_id, reuse) in chronological order; same-date papers share one past.

    Each record has work_id, publication_date and reference_ids. Reference IDs
    are deduplicated within papers. At least 10 linked references are required.
    """
    records=sorted(records,key=lambda p:(p['publication_date'],p['work_id']))
    past=set();start=0
    while start<len(records):
        stop=start+1
        while stop<len(records) and records[stop]['publication_date']==records[start]['publication_date']:stop+=1
        for record in records[start:stop]:
            refs=set(record['reference_ids']);value=len(refs&past)/len(refs) if len(refs)>=10 else np.nan
            yield record['work_id'],value
        for record in records[start:stop]:past.update(record['reference_ids'])
        start=stop


def fixed_library_components(reference_ids,library,previous_reference_ids):
    refs=set(reference_ids);previous=set(previous_reference_ids);library=set(library)
    if len(refs)<10:return dict(fixed_library=np.nan,subsequent_reuse=np.nan,first_observed=np.nan)
    return dict(fixed_library=len(refs&library)/len(refs),subsequent_reuse=len((refs-library)&previous)/len(refs),first_observed=len(refs-library-previous)/len(refs))
