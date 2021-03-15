import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from kaldialign import align

from alignysis.phonetic import BABELCODE2LANG, IGNORED_SYMBOLS, determine_base

GAP_CHAR = '*'


def get_ignored_symbols_re(symbols):
    return re.compile(fr'({r"|".join(symbols)})')


def read_kaldi(text_path) -> Dict[str, str]:
    """
    Reads Kaldi 'text' and '.tra' files.
    'text' is usually found inside Kaldi's data dir.
    '.tra' files are usually found in Kaldi's scoring directory.

    Returns a mapping from an utterance ID to its text.
    """
    id_to_text = dict(
        filter(
            lambda parts: len(parts) == 2,
            (l.split(maxsplit=1) for l in text_path.read_text().splitlines())
        )
    )
    return id_to_text


def read_espnet(text_path) -> Dict[str, str]:
    """
    Reads ESPnet 'hyp.trn' and 'ref.trn' files, typically found inside the 'decode' directory.

    Returns a mapping from an utterance ID to its text.
    """
    id_to_text = {}
    with Path(text_path).open() as f:
        for line in f:
            x = line.strip().rsplit(maxsplit=1)
            if len(x) == 1:
                x = '', x[0]  # empty text
            symbols, id_ = x
            id_to_text[id_[1:-1]] = symbols
    return id_to_text


def join_ref_hyp(id_to_ref: Dict[str, str], id_to_hyp: Dict[str, str]) -> List[Dict[str, str]]:
    # Validation
    ref_ids = set(id_to_ref)
    hyp_ids = set(id_to_hyp)
    common_ids = ref_ids & hyp_ids
    diff = len(ref_ids) - len(common_ids)
    if diff:
        logging.warning(f'Found {diff} missing IDs in the hypothesis set.')
    diff = len(hyp_ids) - len(common_ids)
    if diff:
        logging.warning(f'Found {diff} missing IDs in the reference set (this could be a serious error).')
    # Joining
    sequence_pairs = [{'id': id_, 'ref': id_to_ref[id_], 'hyp': id_to_hyp.get(id_)} for id_ in id_to_ref]
    return sequence_pairs


def compute_confusions(
        sequence_pairs: Iterable[Dict[str, str]],
        lang: str,
        ignore_symbols=None,
        scoring_method='per',
) -> Tuple[pd.DataFrame, ...]:
    """
    Computes the alignments between the true and predicted sequences
    and estimates a confusion matrix between symbols.

    Available scoring methods:
        - 'per' - compute regular phone error rate (e.g. [a:] is scored as a single symbol)
        - 'pter' - compute phone token error rate (e.g. [a:] is scored as two separate symbols /a/ and /:/)
        - 'bper' - compute base phone error rate (e.g. [a:] and [a] are scored as the same symbol because they share the same base phone)

    :param sequence_pairs: Lists of {'id': ..., 'ref': ..., 'hyp': ...}
    :param lang: Language tag.
    :param ignore_symbols: A list of symbols to be removed prior to alignment.
    :param scoring_method: 'per' (default), 'pter' or 'bper'
    :return:
    """
    # Regexp for ignoring symbols
    remove_pattern = get_ignored_symbols_re(ignore_symbols)

    # Compute the alignment
    alis = []
    for item in sequence_pairs:
        id_, ref, hyp = item['id'], item['ref'], item['hyp']
        # Some hyps might be missing
        if hyp is None:
            continue

        # Remove special symbols
        if remove_pattern is not None:
            ref = remove_pattern.sub('', ref)
            hyp = remove_pattern.sub('', hyp)
        else:
            ref = ref.replace(' <silence> ', ' ').replace('<silence>', '').strip()
            hyp = hyp.replace(' <silence> ', ' ').replace('<silence>', '').strip()

        # Convert to lists of phones
        ref, hyp = ref.split(), hyp.split()

        if scoring_method == 'pter':
            ref = list(''.join(ref))
            hyp = list(''.join(hyp))
        elif scoring_method == 'bper':
            # The conditional filters out the symbols for which we do not know the base;
            # This is important for scoring phone token models as they will have tones etc.
            # as separate symbols, and we don't want to mix them up with actual phones
            ref = [determine_base(p) for p in ref if determine_base(p) != '?']
            hyp = [determine_base(p) for p in hyp if determine_base(p) != '?']

        # Align
        ali = align(ref, hyp, GAP_CHAR)
        if not ali:
            continue
        alis.append(ali)

    # Parse the alignments to error types
    errors = defaultdict(lambda: defaultdict(int))
    confusions = defaultdict(lambda: defaultdict(int))
    for ali in alis:
        for ref, hyp in ali:
            assert not (ref == GAP_CHAR and hyp == GAP_CHAR)
            confusions[ref][hyp] += 1
            if ref != GAP_CHAR:
                errors[ref]['TOTAL_TRUE_COUNT'] += 1
            if ref == hyp:
                errors[ref]['OK'] += 1
            elif ref == GAP_CHAR:
                errors[hyp]['INSERTION'] += 1
            elif hyp == GAP_CHAR:
                errors[ref]['DELETION'] += 1
            else:
                errors[ref]['SUBSTITUTION'] += 1

    # make dataframe
    df = pd.DataFrame(errors).T
    df['LANG'] = lang
    confusions_df = pd.DataFrame([
        {'ref': ref, 'hyp': hyp, 'count': count, 'total_ref': errors[ref]['TOTAL_TRUE_COUNT'], 'lang': lang}
        for ref, hyps in confusions.items()
        for hyp, count in hyps.items()
    ])

    return df, confusions_df


def process_asr_results(ref_path: Path, hyp_path: Path, ignored_symbols: Iterable[str], scoring_method='per'):
    """Does everything."""
    print(f'Processing: {ref_path.name} and {hyp_path.name}')

    # Read the data
    if ref_path.suffix == '.trn':
        ref = read_espnet(ref_path)
        hyp = read_espnet(hyp_path)
        lang = ref_path.parent.name.split('_')[2]
        lang = BABELCODE2LANG.get(lang, lang)  # try converting to BABEL
    else:
        ref = read_kaldi(ref_path)
        hyp = read_kaldi(hyp_path)
        lang = ref_path.stem
        lang = BABELCODE2LANG.get(lang, lang)  # try converting to BABEL

    grouped_ref_hyp = join_ref_hyp(id_to_ref=ref, id_to_hyp=hyp)

    df, confusions_df = compute_confusions(grouped_ref_hyp, lang, ignored_symbols, scoring_method)
    return grouped_ref_hyp, lang, df, confusions_df

