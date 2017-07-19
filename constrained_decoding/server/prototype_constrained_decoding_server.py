# coding: utf-8

import json
import codecs

from constrained_decoding.server import DataProcessor, run_imt_server
from constrained_decoding.translation_model.nematus_tm import NematusTranslationModel


def load_config(filename):
    # defaults -- params which are inputs to `nematus/translate.py`, but aren't part of the persisted *.json config
    translate_config = {
        "return_alignment": False
    }
    config = json.loads(codecs.open(filename, encoding='utf8').read())
    return dict(translate_config, **config)


# Working: test with hard-coded config and model paths
# Working: take these as args when calling the start server script
# Working: add pre- and postprocessing to server infrastructure
MODEL_DIR = '/home/chris/Desktop/Dropbox/amunmt_pretrained/src-pe/'
DATADIR = '/home/chris/Desktop/Dropbox/amunmt_pretrained/src-pe/data/'

configs = [
    # '/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe/model.npz.json'
    MODEL_DIR + 'model.npz.json'
]

models = [
    # '/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe/model.4-best.averaged.npz'
    MODEL_DIR + 'model.4-best.averaged.npz'
]

# subword_codes = '/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/data/en.bpe'
en_subword_codes = DATADIR + 'en.bpe'
de_subword_codes = DATADIR + 'de.bpe'

# Make a data processor for this model
# Note: we need different processors for every possible source and target language
en_data_processor = DataProcessor(lang='en', use_subword=True, subword_codes=en_subword_codes)
de_data_processor = DataProcessor(lang='de', use_subword=True, subword_codes=de_subword_codes)

configs = [load_config(f) for f in configs]

# build ensembled TM
nematus_tm = NematusTranslationModel(models, configs, model_weights=None)

model_dict = {('en', 'de'): nematus_tm}
processor_dict = {
                    'en': en_data_processor,
                    'de': de_data_processor
                 }

run_imt_server(models=model_dict, processors=processor_dict)



