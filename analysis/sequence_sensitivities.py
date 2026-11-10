"""First/last publication blocks under the nine reported sample/width settings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.proportion import confint_proportions_2indep,proportions_ztest
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
OUT=ROOT/'results/sequence_sensitivities'
SEED=20261025
def score_difference(success_a, n_a, success_b, n_b):
    lo, hi = confint_proportions_2indep(success_a, n_a, success_b, n_b, method='newcomb')
    z, p = proportions_ztest([success_a, success_b], [n_a, n_b])
    return dict(difference=float(success_a/n_a-success_b/n_b), lower=float(lo), upper=float(hi), z=float(z), p=float(p))

def publication_sequence(pilot=False):
    folder = OUT / ('sequence_pilot' if pilot else 'publication_sequence')
    folder.mkdir(parents=True, exist_ok=True)
    papers = pd.read_parquet(DATA_DIR / 'papers.parquet').merge(pd.read_parquet(DATA_DIR / 'scores/paper_predictions.parquet')[['author_id','work_id','R','expected_calibrated']],on=['author_id','work_id'],validate='one_to_one')
    annual = pd.read_parquet(DATA_DIR / 'scores/author_scores.parquet')
    np.testing.assert_allclose(papers.R, papers.expected_calibrated-papers.alpha.ge(.1))
    papers['publication_date'] = pd.to_datetime(papers.publication_date)
    counts = papers.groupby('author_id').size()
    early = papers[papers.year.le(2024)].groupby('author_id').size()
    late = papers[papers.year.ge(2025)].groupby('author_id').size()
    rows, counts_rows, trajectories = [], [], []
    for minimum in [12, 15, 20]:
        for width in [3, 5, 7]:
            # First/last blocks remain entirely within the two original periods.
            selected = counts.index[counts.ge(max(minimum, 2*width)) & early.ge(width) & late.ge(width)]
            frame = papers[papers.author_id.isin(selected)].sort_values(['author_id', 'publication_date', 'work_id'])
            if pilot:
                selected = pd.Index(selected[:100])
                frame = frame[frame.author_id.isin(selected)]
            valid = []
            excluded_ties = 0
            for author, part in frame.groupby('author_id', sort=False):
                dates = part.publication_date.to_numpy()
                if dates[width-1] == dates[width] or dates[-width] == dates[-width-1]:
                    excluded_ties += 1
                    continue
                first, last = part.iloc[:width], part.iloc[-width:]
                assert first.year.le(2024).all() and last.year.ge(2025).all()
                assert not set(first.work_id).intersection(last.work_id)
                record = dict(author_id=author, total_papers=len(part), early_papers=int(early[author]),
                    late_papers=int(late[author]), minimum_total=minimum, block_width=width,
                    first_R=first.R.mean(), last_R=last.R.mean(),
                    first_alpha=first.alpha.mean(), last_alpha=last.alpha.mean(),
                    first_threshold_positive=first.alpha.ge(.1).mean(), last_threshold_positive=last.alpha.ge(.1).mean(),
                    first_start=str(first.publication_date.min().date()), first_end=str(first.publication_date.max().date()),
                    last_start=str(last.publication_date.min().date()), last_end=str(last.publication_date.max().date()),
                    annual_early_R=float(annual.loc[author, 'early']), annual_late_R=float(annual.loc[author, 'late']))
                valid.append(record)
            data = pd.DataFrame(valid)
            assert len(data) and np.isfinite(data[['first_R','last_R']]).all().all()
            data.to_parquet(folder / f'author_blocks_n{minimum}_w{width}.parquet', index=False)
            counts_rows.append(dict(minimum_total=minimum, block_width=width, eligible_before_ties=len(selected),
                                    excluded_boundary_date_ties=excluded_ties, analyzed_authors=len(data)))
            baseline = data.first_R.gt(0)
            target = data.last_R.gt(0)
            n_a, n_b = int(baseline.sum()), int((~baseline).sum())
            success_a, success_b = int(target[baseline].sum()), int(target[~baseline].sum())
            result = dict(minimum_total=minimum, block_width=width, authors=len(data),
                early_resistant=n_a, early_other=n_b, last_resistant_among_early_resistant=success_a,
                last_resistant_among_early_other=success_b,
                retention=success_a/n_a, later_positive_other=success_b/n_b,
                spearman=float(spearmanr(data.first_R, data.last_R).statistic),
                annual_spearman_same_authors=float(spearmanr(data.annual_early_R, data.annual_late_R).statistic),
                early_sign_agreement_with_annual=float(data.first_R.gt(0).eq(data.annual_early_R.gt(0)).mean()),
                late_sign_agreement_with_annual=float(data.last_R.gt(0).eq(data.annual_late_R.gt(0)).mean()),
                **score_difference(success_a, n_a, success_b, n_b))
            rng = np.random.default_rng(SEED)
            values = data[['first_R', 'last_R']].to_numpy()
            boot = []
            for _ in range(500 if not pilot else 30):
                sampled = values[rng.integers(0, len(values), len(values))]
                boot.append(spearmanr(sampled[:, 0], sampled[:, 1]).statistic)
            result['spearman_lower'], result['spearman_upper'] = np.quantile(boot, [.025, .975])
            rows.append(result)
            print('Sequence sensitivity:', minimum, width, len(data), 'authors', flush=True)
    results = pd.DataFrame(rows)
    # Nine prespecified sensitivity configurations, all retained in the output.
    from statsmodels.stats.multitest import multipletests
    results['holm_p'] = multipletests(results.p, method='holm')[1]
    results.to_csv(folder / 'block_persistence.csv', index=False)
    pd.DataFrame(counts_rows).to_csv(folder / 'sample_flow.csv', index=False)
    common_parts = [pd.read_parquet(folder / f'author_blocks_n15_w{width}.parquet').set_index('author_id') for width in [3, 5, 7]]
    common = common_parts[0].index.intersection(common_parts[1].index).intersection(common_parts[2].index)
    common_rows = []
    for width, part in zip([3, 5, 7], common_parts):
        same = part.loc[common]
        first_positive, last_positive = same.first_R.gt(0), same.last_R.gt(0)
        common_rows.append(dict(width=width, authors=len(same),
            spearman=float(spearmanr(same.first_R, same.last_R).statistic),
            retention=float(last_positive[first_positive].mean()),
            later_positive_other=float(last_positive[~first_positive].mean())))
    pd.DataFrame(common_rows).to_csv(folder / 'same_author_width_sensitivity.csv', index=False)

def run(output=None):
    global OUT
    OUT=Path(output or OUT);publication_sequence()
    actual=pd.read_csv(OUT/'publication_sequence/block_persistence.csv');expected=pd.read_csv(SOURCE_DATA / 'sequence_language/block_persistence.csv')
    cols=['authors','retention','later_positive_other','spearman','lower','upper','spearman_lower','spearman_upper']
    np.testing.assert_allclose(actual[cols],expected[cols],atol=1e-12)
    return {'configurations':len(actual),'all_results_verified':True}
