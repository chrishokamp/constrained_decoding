"""
Implements AbstractConstrainedTM for Nematus NMT models
"""

import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import shared

from nematus.theano_util import (load_params, init_theano_params)
from nematus.nmt import (build_sampler, gen_sample, init_params)
from nematus.compat import fill_options
from nematus.util import load_dict

from . import AbstractConstrainedTM
from .. import ConstraintHypothesis


class NematusTranslationModel(AbstractConstrainedTM):

    def __init__(self, model_files, configs, model_weights=None):
        """"
        Create a ConstrainedTM using Nematus translation models

        Args:
          config: a dict containing key-->value for each argument supported by `nematus/translate.py`

        """

        # WORKING: really we just care that models have the same _output_ vocabulary
        # WORKING: if user specifies a different input per-model in an ensemble, we shouldn't care
        # TODO: it's important that we transparently support weighted ensemble decoding

        assert len(model_files) == len(configs), 'We need config options for each model'

        trng = RandomStreams(1234)
        # don't use noise
        use_noise = shared(numpy.float32(0.))

        if model_weights is None:
            self.model_weights = numpy.ones(len(model_files))
        else:
            assert len(model_weights) == len(model_files), 'if you specify weights, there must be one per model'
            self.model_weights = model_weights

        self.fs_init = []
        self.fs_next = []

        # each entry in self.word_dicts is:
        # `{'input_dicts': [...], 'input_idicts': [...], 'output_dict': <dict>, 'output_idict': <dict>}
        self.word_dicts = []

        for model, config in zip(model_files, configs):
            # fill in any unspecified options in-place
            fill_options(config)
            param_list = numpy.load(model).files
            param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
            params = load_params(model, param_list)
            tparams = init_theano_params(params)

            # load model-specific input and output vocabularies
            # Note: some models have multiple input factors -- if so, we need to split that model's input into factors
            #   using the same logic that was used at training time
            # Note: every model's output vocabulary must be exactly the same in order to do ensemble decoding
            self.word_dicts.append(self.load_dictionaries(config['dictionaries'],
                                                          n_words_src=config.get('n_words_src', None)))



            f_init, f_next = build_sampler(tparams, config, use_noise, trng,
                                           return_alignment=config['return_alignment'])

            self.fs_init.append(f_init)
            self.fs_next.append(f_next)

        # Make sure all output dicts have the same number of items
        assert len(set(len(d['output_dict']) for d in self.word_dicts)) == 1, 'Output vocabularies must be identical'

    @staticmethod
    def load_dictionaries(dictionary_files, n_words_src=None):
        """
        Load the input dictionaries and output dictionary for a model. Note the `n_words_src` kwarg is here to
        maintain compatability with the dictionary loading logic in Nematus.

        Args:
          dictionary_files: list of strings which are paths to *.json Nematus dictionary files

        Returns:
          input_dicts, input_idicts, output_dict, output_idict
        """

        input_dict_files = dictionary_files[:-1]
        output_dict_file = dictionary_files[-1]

        # load source dictionary and invert
        input_dicts = []
        input_idicts = []
        for dictionary in input_dict_files:
            input_dict = load_dict(dictionary)
            if n_words_src is not None:
                for key, idx in input_dict.items():
                    if idx >= n_words_src:
                        del input_dict[key]
            input_idict = dict()
            for kk, vv in input_dict.iteritems():
                input_idict[vv] = kk
            input_idict[0] = '<eos>'
            input_idict[1] = 'UNK'
            input_dicts.append(input_dict)
            input_idicts.append(input_idict)

        # load target dictionary and invert
        output_dict = load_dict(output_dict_file)
        output_idict = dict()
        for kk, vv in output_dict.iteritems():
            output_idict[vv] = kk
        output_idict[0] = '<eos>'
        output_idict[1] = 'UNK'

        return {
            'input_dicts': input_dicts,
            'input_idicts': input_idicts,
            'output_dict': output_dict,
            'output_idict': output_idict
        }

    def start_hypothesis(self, *args, **kwargs):
        pass

    def generate(self, hyp, n_best=1):
        pass

    def generate_constrained(self, hyp):
        pass

    def continue_constrained(self, hyp):
        pass







