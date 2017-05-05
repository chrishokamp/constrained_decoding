"""
Look in the model dir and check dev reports to print paths to the best N models from an experiment

"""

# TODO: see TERM NMT visualization notebook for help

import argparse
import logging
import sys
import json
import codecs
import itertools
import os
import re

import numpy

from constrained_decoding import create_constrained_decoder
from constrained_decoding.translation_model.nematus_tm import NematusTranslationModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_bleu_output(bleu_line):
    '''Parse a line of BLEU score dev report'''
    #BLEU = 46.20, 73.7/51.7/39.2/30.8 (BP=0.998, ratio=0.998, hyp_len=35510, ref_len=35588)
    bleu_s = bleu_line.split()[2]
    # cut off comma
    bleu_s = bleu_s[:-1]
    return float(bleu_s)


def get_all_saved_models(model_dir):
    '''
    Get the absolute paths to all saved models in `model_dir`
      models must be saved as *iter*.npz for this to work.
    '''

    saved_models = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                    if os.path.isfile(os.path.join(model_dir, f)) and re.match('^.*iter.*npz$', f)]

    return saved_models


def best_n_models(model_dir, k_best):
    '''
    Get score report and model list from `model_dir`, rank models by dev performance. 
      Return the absolute paths to the `k_best` models.
      
    We assume filenames follow Nematus naming defaults
    '''

    bleu_score_reports = codecs.open(os.path.join(model_dir,
                                                  'model.npz_bleu_scores'), encoding='utf8').read().strip().split('\n')
    saved_models = get_all_saved_models(model_dir)

    assert len(bleu_score_reports) == len(saved_models), 'There must be 1-to-1 correspondence' + \
                                                         'between models and dev scores for this script to work'

    models_and_scores = zip(saved_models, bleu_score_reports)
    models_and_scores = sorted(models_and_scores, key=lambda x: x[1], reverse=True)

    best_models = models_and_scores[:k_best]
    return best_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_dir', help='Path to experiment model directory', required=True)
    parser.add_argument('-k', '--k_best', type=int, help='The length of the k-best list to return', required=True)
    args = parser.parse_args()

    models, scores = zip(*best_n_models(args.model_dir, args.k_best))

    print('\n'.join(models))

