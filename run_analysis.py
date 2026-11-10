#!/usr/bin/env python3
"""Run manuscript analyses from locally prepared analytical inputs."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['validate','alpha','resistance','prediction','regressions','supplements','tables','all'])
    parser.add_argument('--data-dir',type=Path,help='Local processed publication/author inputs.')
    parser.add_argument('--bootstraps',type=int,default=200,help='Predictive-contribution and coefficient bootstrap samples (publication: 200).')
    parser.add_argument('--score-all',action='store_true',help='In the alpha stage, additionally score all locally supplied abstracts with both reference models.')
    args=parser.parse_args()
    if args.data_dir:os.environ['AI_NONADOPTION_DATA_DIR']=str(args.data_dir.resolve())
    from analysis._config import ROOT, DATA_DIR
    if args.stage=='tables':
        from analysis.export_tables import export
        export(ROOT/'tables')
        return
    if not (DATA_DIR/'papers.parquet').exists():
        parser.error('Supply local inputs with --data-dir /path/to/processed; see README.md.')
    reports={};dest=ROOT/'results';dest.mkdir(exist_ok=True)
    stages=['validate','resistance','prediction','regressions','supplements'] if args.stage=='all' else [args.stage]
    for stage in stages:
        print('Running',stage,flush=True)
        if stage=='validate':
            from analysis.preprocessing import validate
            reports[stage]=validate()
        elif stage=='alpha':
            from alpha.calculate_alpha import build_distribution,verify_examples,score
            reports[stage]={'distributions':[build_distribution(ref,dest/f'alpha/{ref}_word_distribution.parquet') for ref in ['gpt','qwen']],'scoring':verify_examples(dest/'alpha/verified_abstracts.csv')}
            if args.score_all:
                for ref in ['gpt','qwen']:score(DATA_DIR/'abstracts.parquet',dest/f'alpha/{ref}_scores.parquet',ref)
        elif stage=='resistance':
            from analysis.resistance import estimate
            from analysis.persistence import run as persistence
            reports[stage]={'scores':[estimate(.10,dest/'resistance')[2]],'persistence':persistence(dest/'persistence')}
        elif stage=='prediction':
            from analysis.predictive import calculate
            reports[stage]=[calculate(y,a,args.bootstraps,dest/'prediction',complete_coefficients=not a) for a in [False,True] for y in [2025,2026]]
        elif stage=='regressions':
            from analysis.regressions import run
            reports[stage]=run(dest/'regressions')
        elif stage=='supplements':
            from analysis.absolute_alpha import calculate
            from analysis.sequence_language import sequence,next_publication,verify_language
            from analysis.three_groups import run
            from analysis.model_agreement import run as agreement
            from analysis.sequence_sensitivities import run as block_sensitivity
            reports[stage]={'absolute':calculate(dest),'sequence':sequence(dest/'sequence_language'),'block_sensitivity':block_sensitivity(dest/'sequence_sensitivities'),'next_publication':next_publication(dest/'sequence_language'),'language':verify_language(dest/'sequence_language'),'three_groups':run(dest/'three_groups'),'reference_model':agreement(dest/'reference_model')}
        (dest/'analysis_summary.json').write_text(json.dumps(reports,indent=2,default=str))
    print(json.dumps(reports,indent=2,default=str))


if __name__=='__main__':main()
