import html
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit


def initialize_model(path=None):
    if path is None:
        path=Path(__file__).resolve().parent/'alpha_word_distribution.parquet'
    frame = pq.read_table(path)
    human_probability = frame.column('P').to_numpy()
    ai_probability = frame.column('Q').to_numpy()
    weights = np.log(ai_probability)-np.log1p(-ai_probability)-np.log(human_probability)+np.log1p(-human_probability)
    baseline = float(np.log1p(-ai_probability).sum()-np.log1p(-human_probability).sum())
    return dict(zip(frame.column('Word').to_pylist(), weights)), baseline


def score_alpha(text, model):
    weights, baseline = model
    text = html.unescape(text).replace('\r', ' ').replace('\x0c', ' ')
    text = ' '.join(re.sub(r'<[^>]+>', ' ', text).split())
    likelihood_ratios = []
    matched = 0
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
        if not sentence.strip():
            continue
        words = {word for word in re.findall(r'\b\w+\b', sentence.lower()) if not word.isdigit()}
        selected = words.intersection(weights)
        matched += len(selected)
        likelihood_ratios.append(baseline+sum(weights[word] for word in selected))
    if not matched or not likelihood_ratios:
        return None, 'low_information'
    ratios = np.asarray(likelihood_ratios)
    if np.mean(np.expm1(np.clip(ratios, -700, 700))) <= 0:
        return 0.0, 'scored'
    if np.mean(-np.expm1(np.clip(-ratios, -700, 700))) >= 0:
        return 1.0, 'scored'
    lower, upper = 0.0, 1.0
    for iteration in range(32):
        midpoint = (lower+upper)/2
        if expit(math.log(midpoint)-math.log1p(-midpoint)+ratios).mean() > midpoint:
            lower = midpoint
        else:
            upper = midpoint
    return (lower+upper)/2, 'scored'
