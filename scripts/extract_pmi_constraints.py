"""
Extract a PMI terminology from a parallel dataset
"""
from __future__ import print_function

import logging
import argparse
import codecs
import errno
import json
import os
from collections import Counter, OrderedDict, defaultdict

import numpy

from semantic_annotator.spotting import MatchSpotter


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def normalized_pmi(p_x, p_y, p_y_given_x):
    joint = p_x * p_y_given_x
    normalize = -numpy.log(joint)
    return numpy.log(p_y_given_x / p_y) / normalize


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_segments_from_file(filename, max_segs=100000):
    with codecs.open(filename, encoding='utf8') as inp:
        for i, l in enumerate(inp):
            if i < max_segs:
                yield l.strip().split()
            else:
                raise StopIteration


def all_ngrams_in_segment(segment, min_ngram, max_ngram):
    # note we return the set to only count each ngram once per segment
    return list(set([u' '.join(ngram) for factor in range(min_ngram, max_ngram+1)
                     for ngram in [segment[i:i+factor] for i in range(len(segment)-factor+1)]]))


def count_ngrams_in_corpus(segments, min_ngram, max_ngram):
    ngram_counts = Counter()
    ngram_lines = []
    for segment in segments:
        ngrams = all_ngrams_in_segment(segment, min_ngram, max_ngram)
        ngram_counts.update(ngrams)
        ngram_lines.append(ngrams)
    return ngram_counts, ngram_lines


# Note: side effect function
def prune_high_low_freq(counter, min_occs, max_occs):
    for ngram, count in counter.items():
        if count > max_occs or count < min_occs:
            del counter[ngram]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--train_source", required=True,
                        help="the source text for the training corpus")
    parser.add_argument("-t", "--train_target", required=True,
                        help="the target text for the training corpus")
    parser.add_argument("--test_source", required=True,
                        help="the source data for the test set -- constraints will be extracted for the target side.")
    parser.add_argument("--min_ngram", type=int, default=2,
                        help="the minimum ngram length to extract")
    parser.add_argument("--max_ngram", type=int, default=5,
                        help="the max ngram length to extract")
    parser.add_argument("--num_segments", type=int, default=100000,
                        help="the maximum number of segments to use from the training data")
    parser.add_argument("--min_occs", type=int, default=5,
                        help="the minimum number of occurences for a phrase to be considered")
    parser.add_argument("--max_occs", type=int, default=500,
                        help="the maximum number of occurences for a phrase to be considered")
    parser.add_argument("-o", "--outputdir", required=True,
                        help="the directory where we should write the test set constraints")

    args = parser.parse_args()

    train_source = get_segments_from_file(args.train_source, max_segs=args.num_segments)
    train_target = get_segments_from_file(args.train_target, max_segs=args.num_segments)

    source_ngrams, source_line_ngrams = count_ngrams_in_corpus(train_source,
                                                                min_ngram=args.min_ngram,
                                                                max_ngram=args.max_ngram)
    target_ngrams, target_line_ngrams = count_ngrams_in_corpus(train_target,
                                                                min_ngram=args.min_ngram,
                                                                max_ngram=args.max_ngram)

    num_source_segments = len(source_line_ngrams)
    num_target_segments = len(target_line_ngrams)
    assert num_source_segments == num_target_segments, 'source and target files must have the same # of segments'

    logger.info('Ngrams in source: {}'.format(len(source_ngrams)))
    logger.info('Ngrams in target: {}'.format(len(target_ngrams)))

    # Note in place modification of dictionaries
    prune_high_low_freq(source_ngrams, args.min_occs, args.max_occs)
    prune_high_low_freq(target_ngrams, args.min_occs, args.max_occs)

    logger.info('After pruning: Ngrams in source: {}'.format(len(source_ngrams)))
    logger.info('After pruning: Ngrams in target: {}'.format(len(target_ngrams)))

    # compute priors
    src_priors = OrderedDict((k, v / float(num_source_segments)) for k, v in source_ngrams.most_common())
    trg_priors = OrderedDict((k, v / float(num_source_segments)) for k, v in target_ngrams.most_common())

    chunk_map = defaultdict(Counter)

    for src_chunks, trg_chunks in zip(source_line_ngrams, target_line_ngrams):
        for src_chunk in set(src_chunks):
            if src_chunk in source_ngrams:
                chunk_map[src_chunk].update(list(set([chunk for chunk in trg_chunks if chunk in target_ngrams])))

    # optionally free up some memory
    # del source_line_ngrams
    # del target_line_ngrams

    src_posteriors = {}
    num_processed = 0
    for src_chunk, trg_chunk_counter in chunk_map.items():
        total_occs = float(source_ngrams[src_chunk])
        posteriors = OrderedDict([(k, v / total_occs) for k, v in trg_chunk_counter.most_common()])
        # optionally free memory
        # del chunk_map[src_chunk]
        src_posteriors[src_chunk] = posteriors
        num_processed += 1
        if num_processed % 10000 == 0:
            logger.info('Computing posteriors: processed {}'.format(num_processed))

    # optionally free memory
    # del chunk_map

    pmi_cands = {}

    num_processed = 0
    for source_phrase, posteriors in src_posteriors.items():
        source_prior = src_priors[source_phrase]
        pmi_scores = []
        for target_phrase, posterior in posteriors.items():
            target_prior = trg_priors[target_phrase]
            pmi_score = normalized_pmi(source_prior, target_prior, posterior)
            pmi_scores.append((target_phrase, pmi_score))

        pmi_scores = sorted(pmi_scores, key=lambda x: x[1], reverse=True)
        pmi_cands[source_phrase] = pmi_scores
        #     del src_posteriors[source_phrase]
        num_processed += 1
        if num_processed % 10000 == 0:
            print('Computing PMI scores: processed {}'.format(num_processed))


    # filter the map to only pairs which are likely to be good
    # TODO: these should be hyperparams
    min_occs = args.min_occs
    min_score = 0.95
    min_source_len = 5

    good_cands = []
    for src, cands in pmi_cands.items():
        if len(src) >= min_source_len and len(cands) > 0 and cands[0][1] >= min_score:
            if source_ngrams[src] > min_occs:
                #             print(u'match: {}-->{} score {}'.format(src, cands[0][0], cands[0][1]))
                internal_cands = [cand for cand in cands
                                  if target_ngrams[cand[0]] > min_occs
                                  and cand[1] >= min_score]
                # sort by length descending
                internal_cands = sorted(internal_cands, key=lambda x: len(x[0].split()), reverse=True)
                if len(internal_cands) > 0:
                    good_cands.append((src, cands[0]))


    # extract source language rules for the spotter
    src_rules = [src for src, (trg, score) in good_cands]
    term_pair_map = OrderedDict((k, v) for k, (v, s) in good_cands)

    term_spotter = MatchSpotter(rules=src_rules)

    # Note that source, target and test_source should all already be tokenized and encoded
    prepped_dev_lines = codecs.open(args.test_source, encoding='utf8').read().strip().split('\n')

    # Get all the spots
    dev_term_spots = []
    for l in prepped_dev_lines:
        spots = term_spotter.get_spots(l)
        dev_term_spots.append(spots)

    dev_term_constraints = []
    for text, spots in zip(prepped_dev_lines, dev_term_spots):
        output_constraints = []
        if len(spots) > 0:
            for spot in spots:
                if spot[1] - spot[0] > 1:
                    spotted_term = text[spot[0]:spot[1]]
                    mapped_term = term_pair_map[spotted_term]
                    output_constraints.append(mapped_term.split())
        dev_term_constraints.append(output_constraints)

    num_constraints = len([c for l in dev_term_constraints for c in l])

    # Write out the test set target side constraints to the user-specified directory
    output_file_name = 'pmi.constraints.json'

    with codecs.open(os.path.join(args.outputdir, output_file_name), 'w', encoding='utf8') as out:
        out.write(json.dumps(dev_term_constraints, indent=2))

    logger.info('Wrote {} constraints to {}'.format(num_constraints, os.path.join(args.outputdir, output_file_name)))


