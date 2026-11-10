"""Agreement of abstract scores under GPT and Qwen reference corpora."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
SEED=20261026
def agreement(x, y, threshold=.1):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = x >= threshold, y >= threshold
    nn, np_, pn, pp = [int(v.sum()) for v in [~a & ~b, ~a & b, a & ~b, a & b]]
    observed = float(np.mean(a == b))
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return dict(threshold=threshold, agreement=observed,
        cohen_kappa=(observed - expected) / (1 - expected) if expected < 1 else None,
        gpt_at_or_above=float(a.mean()), qwen_at_or_above=float(b.mean()),
        gpt_below_qwen_below=nn, gpt_below_qwen_above=np_, gpt_above_qwen_below=pn, gpt_above_qwen_above=pp,
        positive_agreement=2 * pp / (2 * pp + np_ + pn) if (2 * pp + np_ + pn) else None,
        negative_agreement=2 * nn / (2 * nn + np_ + pn) if (2 * nn + np_ + pn) else None)

def continuous(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    covariance = np.mean((x - mx) * (y - my))
    vx, vy = x.var(), y.var()
    return dict(gpt_mean=float(mx), qwen_mean=float(my),
        pearson=float(covariance / np.sqrt(vx * vy)), spearman=float(spearmanr(x, y).statistic),
        lin_concordance=float(2 * covariance / (vx + vy + (mx - my) ** 2)),
        mean_difference_qwen_minus_gpt=float((y - x).mean()), mean_absolute_difference=float(np.abs(y - x).mean()),
        root_mean_square_difference=float(np.sqrt(np.mean((y - x) ** 2))))

def bootstrap_metrics(frame, n_boot=400):
    x, y = frame.alpha_gpt.to_numpy(), frame.alpha_qwen.to_numpy()
    a, b = x >= .1, y >= .1
    aggregates = pd.DataFrame(dict(author_id=frame.author_id.to_numpy(), n=1., x=x, y=y, xx=x*x, yy=y*y,
        xy=x*y, diff=y-x, absolute=np.abs(y-x), same=(a==b).astype(float), gpt=a.astype(float), qwen=b.astype(float)))
    matrix = aggregates.groupby('author_id', sort=True).sum().to_numpy()
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(0, len(matrix), len(matrix)), minlength=len(matrix))
        n, sx, sy, sxx, syy, sxy, difference, absolute, same, ga, qa = weights @ matrix
        mx, my = sx/n, sy/n
        vx, vy, cov = sxx/n-mx*mx, syy/n-my*my, sxy/n-mx*my
        exp_agree = (ga/n)*(qa/n)+(1-ga/n)*(1-qa/n)
        samples.append([cov/np.sqrt(vx*vy), 2*cov/(vx+vy+(mx-my)**2), difference/n, absolute/n,
                        same/n, (same/n-exp_agree)/(1-exp_agree)])
    bounds = np.quantile(np.asarray(samples), [.025, .975], axis=0)
    names = ['pearson', 'lin_concordance', 'mean_difference_qwen_minus_gpt', 'mean_absolute_difference', 'agreement', 'cohen_kappa']
    return {name: dict(lower=float(bounds[0, j]), upper=float(bounds[1, j])) for j, name in enumerate(names)}

def run(output=None,bootstraps=400):
    data=pd.read_parquet(DATA_DIR / 'reference_model/paper_scores.parquet')
    dest=Path(output or ROOT/'results/reference_model');dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for period,part in [('All',data)]+[(str(y),p) for y,p in data.groupby('year')]:
        rows.append(dict(unit='Paper',period=period,observations=len(part),authors=part.author_id.nunique(),**continuous(part.alpha_gpt,part.alpha_qwen),**agreement(part.alpha_gpt,part.alpha_qwen)))
    result=pd.DataFrame(rows);expected=pd.read_csv(DATA_DIR / 'reference_model/agreement.csv',dtype={'period':str});expected=expected[expected.unit.eq('Paper')].iloc[:len(rows)]
    cols=['gpt_mean','qwen_mean','pearson','spearman','agreement','cohen_kappa']
    np.testing.assert_allclose(result[cols],expected[cols],atol=1e-12)
    result.to_csv(dest/'paper_agreement.csv',index=False)
    edges=np.linspace(0,1,21);bins=np.minimum(np.searchsorted(edges,data.alpha_gpt,side='right')-1,19)
    author,unique=pd.factorize(data.author_id,sort=True);flat=author*20+bins
    counts=np.bincount(flat,minlength=len(unique)*20).reshape(len(unique),20).astype(float)
    sums=np.bincount(flat,weights=data.alpha_qwen,minlength=len(unique)*20).reshape(len(unique),20)
    n=counts.sum(axis=0);x=np.bincount(bins,weights=data.alpha_gpt,minlength=20)/n;y=sums.sum(axis=0)/n
    rng=np.random.default_rng(SEED);boot=[]
    for _ in range(bootstraps):
        w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique));boot.append((w@sums)/(w@counts))
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    summary=pd.DataFrame(dict(GPT_bin_left=edges[:-1],GPT_bin_right=edges[1:],papers=n.astype(int),mean_GPT_alpha=x,mean_Qwen_alpha=y,lower_95=lo,upper_95=hi))
    if bootstraps==400:np.testing.assert_allclose(summary,pd.read_csv(SOURCE_DATA / 'qwen_gpt_binned_scores.csv'),atol=1e-12)
    summary.to_csv(dest/'binned_scores.csv',index=False)
    return dict(papers=len(data),authors=len(unique),Pearson=rows[0]['pearson'],Spearman=rows[0]['spearman'],bootstrap_samples=bootstraps,agreement_verified=True)
