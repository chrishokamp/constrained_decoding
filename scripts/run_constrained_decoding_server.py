# coding: utf-8

import json
import codecs
import argparse

from constrained_decoding.server import DataProcessor
from constrained_decoding.server.nmt_app import run_imt_server
from constrained_decoding.translation_model.nematus_tm import NematusTranslationModel


def load_config(filename):
    # defaults -- params which are inputs to `nematus/translate.py`, but aren't part of the persisted *.json config
    translate_config = {
        "return_alignment": False
    }
    config = json.loads(codecs.open(filename, encoding='utf8').read())
    return dict(translate_config, **config)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+', help='the paths to one or more Nematus models', required=True)
    parser.add_argument('-c', '--configs', nargs='+', help='paths to one or more config.json files for Nematus models',
                        default=None, required=False)

    # Note: using only one source subword dict stops us from using multiple models with different inputs
    parser.add_argument('--source_subword_codes', help='path to the source subword codes', required=True)
    parser.add_argument('--target_subword_codes', help='path to the target subword codes', required=True)

    parser.add_argument('--source_lang', type=str, help='two-character source language code', required=True)
    parser.add_argument('--target_lang', type=str, help='two-character target language code', required=True)

    parser.add_argument('--source_truecase', default=None, type=str,
                        help='(Optional) Path to the source truecasing model',
                        required=False)
    parser.add_argument('--target_truecase', default=None, type=str,
                        help='(Optional) Path to the target truecasing model',
                        required=False)
    parser.add_argument('--escape_special_chars', dest='escape_special_chars', action='store_true',
                        help='(Optional) if --escape_special_chars, we will map special punctuation to html entities')
    parser.set_defaults(escape_special_chars=False)
    args = parser.parse_args()

    assert len(args.models) == len(args.configs), 'Number of models differs from numer of config files'
    assert len(args.source_lang) == len(args.target_lang) == 2, 'Language codes must be two characters'

    # Make a data processor for this model
    # Note: we need different processors for every possible source and target language
    src_data_processor = DataProcessor(lang=args.source_lang, use_subword=True,
                                       subword_codes=args.source_subword_codes,
                                       truecase_model=args.source_truecase,
                                       escape_special_chars=args.escape_special_chars)
    trg_data_processor = DataProcessor(lang=args.target_lang, use_subword=True,
                                       subword_codes=args.target_subword_codes,
                                       truecase_model=args.target_truecase,
                                       escape_special_chars=args.escape_special_chars)

    configs = [load_config(f) for f in args.configs]

    # build ensembled TM
    nematus_tm = NematusTranslationModel(args.models, configs, model_weights=None)

    model_dict = {(args.source_lang, args.target_lang): nematus_tm}
    processor_dict = {
        args.source_lang: src_data_processor,
        args.target_lang: trg_data_processor
    }

    run_imt_server(models=model_dict, processors=processor_dict)



