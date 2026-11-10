"""Rebuild human/AI word distributions and calculate abstract scores."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from alpha.reference_tools import sentence_counts,batch_alpha
from alpha.scoring import initialize_model
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA


def distribution_path(reference='gpt'):
    name='alpha_word_distribution.parquet' if reference=='gpt' else 'qwen_word_distribution.parquet'
    bundled=Path(__file__).resolve().parent/name
    return bundled if bundled.is_file() else DATA_DIR/'alpha'/name


def build_distribution(reference='gpt',output=None):
    pairs=pd.read_parquet(DATA_DIR/f'alpha/human_{reference}_reference_pairs.parquet')
    human,nh=sentence_counts(pairs.abstract_text);ai,na=sentence_counts(pairs.rewritten_abstract)
    words=sorted(w for w,c in human.items() if c>=5 and ai.get(w,0)>=3)
    data=pd.DataFrame({'Word':words,'P':[(human[w]+.5)/(nh+1) for w in words],'Q':[(ai[w]+.5)/(na+1) for w in words]})
    stored=distribution_path(reference)
    expected=pd.read_parquet(stored)[['Word','P','Q']].sort_values('Word').reset_index(drop=True)
    assert data.Word.equals(expected.Word);np.testing.assert_allclose(data[['P','Q']],expected[['P','Q']],atol=1e-15)
    if output is not None:Path(output).parent.mkdir(parents=True,exist_ok=True);data.to_parquet(output,index=False)
    return dict(reference=reference,pairs=len(pairs),human_sentences=nh,ai_sentences=na,vocabulary=len(words),distribution_verified=True)


def score(input_path,output_path,reference='gpt',batch_size=2000):
    data=pd.read_parquet(input_path);assert {'work_id','abstract_text'}<=set(data)
    model=initialize_model(distribution_path(reference))
    pieces=[]
    for start in range(0,len(data),batch_size):
        block=data.iloc[start:start+batch_size];a,n,m=batch_alpha(block.abstract_text,model)
        pieces.append(pd.DataFrame(dict(work_id=block.work_id.to_numpy(),alpha=a,sentence_count=n,matched_word_occurrences=m)))
    result=pd.concat(pieces,ignore_index=True);Path(output_path).parent.mkdir(parents=True,exist_ok=True);result.to_parquet(output_path,index=False)
    return result


def verify_examples(output=None):
    abstracts=pd.read_parquet(DATA_DIR / 'abstracts.parquet');paired=pd.read_parquet(DATA_DIR / 'reference_model/paper_scores.parquet')
    selected=paired.groupby(['year','field'],observed=True).head(2).merge(abstracts,on='work_id',validate='one_to_one')
    records=selected[['work_id','author_id','year','field']].copy()
    for ref in ['gpt','qwen']:
        model=initialize_model(distribution_path(ref))
        a,n,m=batch_alpha(selected.abstract_text,model)
        np.testing.assert_allclose(a,selected['alpha_'+ref],atol=1e-10)
        records['alpha_'+ref]=a
    if output is not None:Path(output).parent.mkdir(parents=True,exist_ok=True);records.to_csv(output,index=False)
    return {'abstracts_verified':len(records),'both_reference_models_verified':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference',choices=['gpt','qwen'],default='gpt');parser.add_argument('--build-distribution',action='store_true');parser.add_argument('--input');parser.add_argument('--output');args=parser.parse_args()
    if args.build_distribution:print(json.dumps(build_distribution(args.reference,args.output),indent=2))
    elif args.input and args.output:score(args.input,args.output,args.reference)
    else:parser.print_help()
