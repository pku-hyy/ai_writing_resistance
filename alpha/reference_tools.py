"""Reference-text preprocessing and vocabulary construction."""
from collections import Counter
import html
import re
import numpy as np
import pandas as pd
from scipy.special import expit
SPLIT=re.compile(r'(?<=[.!?])\s+|\n+')
WORD=re.compile(r'\b\w+\b')
def sentences(text):
    text = html.unescape(str(text or '')).replace('\r', ' ').replace('\x0c', ' ')
    text = ' '.join(re.sub(r'<[^>]+>', ' ', text).split())
    return [s for s in SPLIT.split(text) if s.strip()]

def tokens(text):
    return {w for w in WORD.findall(text.lower()) if not w.isdigit()}

def sentence_counts(texts):
    counts = Counter()
    n = 0
    for text in texts:
        for sentence in sentences(text):
            counts.update(tokens(sentence))
            n += 1
    return counts, n

def batch_alpha(texts, model):
    weights, baseline = model
    likelihoods, owners, sentence_n, matched_n = [], [], [], []
    for owner, text in enumerate(texts):
        sents = sentences(text)
        matched = 0
        for sentence in sents:
            selected = tokens(sentence).intersection(weights)
            matched += len(selected)
            likelihoods.append(baseline + sum(weights[w] for w in selected))
            owners.append(owner)
        sentence_n.append(len(sents))
        matched_n.append(matched)
    lr = np.asarray(likelihoods, dtype=float)
    owner = np.asarray(owners, dtype=np.int64)
    n = np.asarray(sentence_n, dtype=float)
    count = len(n)
    denominator = np.maximum(n, 1)
    low, high = np.zeros(count), np.ones(count)
    left = np.bincount(owner, weights=np.expm1(np.clip(lr, -700, 700)), minlength=count) / denominator
    right = np.bincount(owner, weights=-np.expm1(np.clip(-lr, -700, 700)), minlength=count) / denominator
    for _ in range(32):
        middle = (low + high) / 2
        posterior = expit((np.log(middle) - np.log1p(-middle))[owner] + lr)
        mean = np.bincount(owner, weights=posterior, minlength=count) / denominator
        up = mean > middle
        low = np.where(up, middle, low)
        high = np.where(up, high, middle)
    result = (low + high) / 2
    result[left <= 0] = 0
    result[right >= 0] = 1
    matched_n = np.asarray(matched_n, dtype=np.int64)
    result[(n == 0) | (matched_n == 0)] = np.nan
    return result, n.astype(np.int64), matched_n
