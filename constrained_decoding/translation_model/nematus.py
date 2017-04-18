"""
Implements AbstractConstrainedTM for Nematus NMT models
"""

import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import shared

from nematus.theano_util import (load_params, init_theano_params)
from nematus.nmt import (build_sampler, gen_sample, init_params)

from . import AbstractConstrainedTM
from .. import ConstraintHypothesis




class NematusTranslationModel(AbstractConstrainedTM):

    def __init__(self, config):
        """"
        Create a ConstrainedTM using Nematus translation models

        Args:
          config: a dict containing key-->value for each argument supported by `nematus/translate.py`

        """

        # TODO: model option loading here: https://github.com/rsennrich/nematus/blob/master/nematus/translate.py#L109-L118
        assert len(config['models']) == len(config['options']), 'We need config options for each model'
        models = config['models']
        options = config['options']

        trng = RandomStreams(1234)
        use_noise = shared(numpy.float32(0.))

        self.fs_init = []
        self.fs_next = []

        for model, option in zip(models, options):
            param_list = numpy.load(model).files
            param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
            params = load_params(model, param_list)
            tparams = init_theano_params(params)

            # word index
            f_init, f_next = build_sampler(tparams, option, use_noise, trng, return_alignment=config['return_alignment'])

            self.fs_init.append(f_init)
            self.fs_next.append(f_next)

    def start_hypothesis(self, *args, **kwargs):
        pass

    def generate(self, hyp, n_best=1):
        pass

    def generate_constrained(self, hyp):
        pass

    def continue_constrained(self, hyp):
        pass







