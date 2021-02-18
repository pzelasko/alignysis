import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from functools import partial

import pandas as pd

from alignysis import IGNORED_SYMBOLS, determine_manner, determine_place, get_special_symbols, process_asr_results
from alignysis.logging import setup_logger


def read_e2e(root: Path, tag: str, num_jobs: int = 8, executor_type=ProcessPoolExecutor):
    # In E2E experiments we read the ground truth from mono experiments reference text
    ground_truth_paths = sorted(p for p in (root / 'mono').rglob('ref.trn'))
    special_symbols = get_special_symbols(ground_truth_paths)
    special_symbols |= set(IGNORED_SYMBOLS)

    expt_dfs = {}
    expt_confusions = {}
    # We use the "subexpt" for consistency with hybrid expts, as they have multiple LMs for each AM
    subexpt = None
    with executor_type(num_jobs) as ex:
        for scoring_method in ['per', 'pter', 'bper']:
            for expt in ['mono', 'multi', 'cross']:
                logging.info(f'[E2E] Reading {expt} expts with {scoring_method.upper()} scoring.')
                exp_root = root / expt
                results_paths = sorted(exp_root.rglob('hyp.trn'))
                # Wrap into partial for easier parallelism
                work = partial(
                    process_asr_results,
                    ignored_symbols=special_symbols,
                    scoring_method=scoring_method
                )
                lang_dfs, lang_confusions = zip(*list(ex.map(work, ground_truth_paths, results_paths)))
                key = (tag, expt, subexpt, scoring_method)
                expt_dfs[key] = lang_dfs
                expt_confusions[key] = lang_confusions
    return expt_dfs, expt_confusions


def read_hybrid(root: Path, tag: str, num_jobs: int = 8, executor_type=ProcessPoolExecutor):
    # We assume that the ground truth texts are copied from a Kaldi data dir and have a .txt suffix.
    ground_truth_paths = sorted(p for p in (root / 'ref_text').glob('*.txt'))
    special_symbols = get_special_symbols(ground_truth_paths)
    special_symbols |= set(IGNORED_SYMBOLS)

    ex = executor_type(num_jobs)

    expt_dfs = {}
    expt_confusions = {}
    with ex:
        for scoring_method in ['per', 'pter', 'bper']:
            for expt in ['mono', 'multi', 'cross']:
                exp_root = root / f'{expt}_tdnnf'
                for subexpt_path in exp_root.glob('*'):
                    subexpt = subexpt_path.name
                    logging.info(f'[Hybrid] Reading {expt}/{subexpt} expts with {scoring_method.upper()} scoring.')
                    results_paths = sorted((root / f'{expt}_tdnnf/{subexpt}').rglob('*.tra'))
                    # Wrap into partial for easier parallelism
                    work = partial(
                        process_asr_results,
                        ignored_symbols=special_symbols,
                        scoring_method=scoring_method
                    )
                    lang_dfs, lang_confusions = zip(*list(ex.map(work, ground_truth_paths, results_paths)))
                    key = (tag, expt, subexpt, scoring_method)
                    expt_dfs[key] = lang_dfs
                    expt_confusions[key] = lang_confusions
    return expt_dfs, expt_confusions


def read_all_expts(root: Path = Path('/Users/pzelasko/jhu/discophone')):
    expt_dfs = {}
    expt_confusions = {}

    e2e_phone_token_root = root / 'discophone-is2020-results-for-journal'
    o1, o2 = read_e2e(e2e_phone_token_root, tag='e2e_phonetokens')
    expt_dfs.update(o1)
    expt_confusions.update(o2)

    e2e_phones_root = root / 'discophone-journal-results'
    o1, o2 = read_e2e(e2e_phones_root, tag='e2e_phones')
    expt_dfs.update(o1)
    expt_confusions.update(o2)

    hybrid_phones_root = root / 'discophone-hybrid-results'
    o1, o2 = read_hybrid(hybrid_phones_root, tag='hybrid_phones')
    expt_dfs.update(o1)
    expt_confusions.update(o2)

    return expt_dfs, expt_confusions


def aggregate_confusions(conf_dfs):
    conf_merged_dfs = {
        ex: pd.concat(lang_dfs)
        for ex, lang_dfs in conf_dfs.items()
    }
    # Filtering ignored symbols - try not to do it for now and see what happens
    # conf_merged_dfs = {
    #     ex: df[(~df.ref.isin(IGNORED_SYMBOLS) & ~df.hyp.isin(IGNORED_SYMBOLS))]
    #     for ex, df in conf_merged_dfs.items()
    # }

    to_concat = []
    for (tag, AM, LM, tok_type), cdf in conf_merged_dfs.items():
        cdf['SYSTEM'] = tag
        cdf['AM'] = AM
        cdf['LM'] = LM
        cdf['TOKEN_TYPE'] = tok_type
        cdf['EXP'] = f'{tag}_{AM}_{LM}_{tok_type}'
        to_concat.append(cdf)

    df = pd.concat(to_concat)
    # df = df[~df.token.str.contains('<')]
    return df


def aggregate_expts(expt_dfs):
    expt_dfs = {
        ex: [
            d.reset_index() for d in lang_dfs
        ] for ex, lang_dfs in expt_dfs.items()
    }
    expt_merged_dfs = {
        ex: pd
            .concat(lang_dfs)
            .rename({'index': 'token'}, axis='columns')
        for ex, lang_dfs in expt_dfs.items()
    }

    expt_merged_dfs = {
        ex: df[~df.token.isin(IGNORED_SYMBOLS)]
        for ex, df in expt_merged_dfs.items()
    }

    to_concat = []
    for (tag, AM, LM, tok_type), edf in expt_merged_dfs.items():
        edf['SYSTEM'] = tag
        edf['AM'] = AM
        edf['LM'] = LM
        edf['TOKEN_TYPE'] = tok_type
        edf['EXP'] = f'{tag}_{AM}_{LM}_{tok_type}'
        to_concat.append(edf)

    df = pd.concat(to_concat)

    df.loc[~df.TOTAL_TRUE_COUNT.isna() & df.OK.isna(), 'OK'] = 0.0
    df.loc[~df.TOTAL_TRUE_COUNT.isna() & df.DELETION.isna(), 'DELETION'] = 0.0
    df.loc[~df.TOTAL_TRUE_COUNT.isna() & df.SUBSTITUTION.isna(), 'SUBSTITUTION'] = 0.0
    df.loc[~df.TOTAL_TRUE_COUNT.isna() & df.INSERTION.isna(), 'INSERTION'] = 0.0

    df = df[~df.token.str.contains('<')]

    return df


def run_data_prep(force: bool = False):
    setup_logger()
    AGG_PATH = Path('art/agg_df.pkl')
    AGG_CONF_PATH = Path('art/agg_conf_df.pkl')
    if AGG_PATH.exists() and AGG_CONF_PATH.exists() and not force:
        logging.info('Reading cached aggregated DFs.')
        return pickle.load(AGG_PATH.open('rb')), pickle.load(AGG_CONF_PATH.open('rb'))

    EXPT_PATH = Path('art/expt_dfs.pkl')
    CONF_PATH = Path('art/expt_confusions.pkl')
    if EXPT_PATH.exists() and CONF_PATH.exists() and not force:
        logging.info('Reading cached per-experiment DFs.')
        dfs = pickle.load(EXPT_PATH.open('rb'))
        confs = pickle.load(CONF_PATH.open('rb'))
    else:
        logging.info('Computing alignments and confusions from raw results.')
        dfs, confs = read_all_expts()
        pickle.dump(dfs, open(EXPT_PATH, 'wb'))
        pickle.dump(confs, open(CONF_PATH, 'wb'))

    if AGG_PATH.exists() and not force:
        logging.info('Reading cached aggregated alignment DF.')
        df = pickle.load(AGG_PATH.open('rb'))
    else:
        logging.info('Aggregating per-experiment alignments.')
        df = aggregate_expts(dfs)
        pickle.dump(df, open(AGG_PATH, 'wb'))

    if AGG_CONF_PATH.exists() and not force:
        logging.info('Reading cached aggregated confusions DF.')
        conf_df = pickle.load(AGG_CONF_PATH.open('rb'))
    else:
        logging.info('Aggregating per-experiment confusions.')
        conf_df = aggregate_confusions(confs)
        conf_df['ref_place'] = conf_df.ref.apply(lambda phn: determine_place(phn))
        conf_df['hyp_place'] = conf_df.hyp.apply(lambda phn: determine_place(phn))
        conf_df['ref_manner'] = conf_df.ref.apply(lambda phn: determine_manner(phn))
        conf_df['hyp_manner'] = conf_df.hyp.apply(lambda phn: determine_manner(phn))
        pickle.dump(conf_df, open(AGG_CONF_PATH, 'wb'))

    logging.info('DF prep finished!')
    return df, conf_df
