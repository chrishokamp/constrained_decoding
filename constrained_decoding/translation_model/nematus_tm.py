"""
Implements AbstractConstrainedTM for Nematus NMT models
"""

import copy

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

        self.eos_token = '<eos>'

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
                                                          n_words_src=config.get('n_words_src', None),
                                                          n_words_trg=config.get('n_words', None)))

            f_init, f_next = build_sampler(tparams, config, use_noise, trng,
                                           return_alignment=config['return_alignment'])

            self.fs_init.append(f_init)
            self.fs_next.append(f_next)

        # Make sure all output dicts have the same number of items
        assert len(set(len(d['output_dict']) for d in self.word_dicts)) == 1, 'Output vocabularies must be identical'

        self.num_models = len(self.fs_init)

        if model_weights is None:
            self.model_weights = numpy.ones(len(model_files))
        else:
            assert len(model_weights) == len(model_files), 'if you specify weights, there must be one per model'
            self.model_weights = numpy.array(model_weights)


    @staticmethod
    def load_dictionaries(dictionary_files, n_words_src=None, n_words_trg=None):
        """
        Load the input dictionaries and output dictionary for a model. Note the `n_words_src` kwarg is here to
        maintain compatibility with the dictionary loading logic in Nematus.

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
        if n_words_trg is not None:
            for key, idx in output_dict.items():
                if idx >= n_words_trg:
                    del output_dict[key]
        output_idict = dict()
        for kk, vv in output_dict.iteritems():
            output_idict[vv] = kk
        output_idict[0] = '<eos>'
        output_idict[1] = 'UNK'

        return {
            'input_dicts': input_dicts,
            'input_idicts': input_idicts,
            'output_dict': output_dict,
            'output_idict': output_idict,
            'src_size': n_words_src,
            'trg_size': n_words_trg
        }

    def map_inputs(self, inputs, factor_separator='|'):
        """
        Map inputs to sequences of ints, which are token indices for the embedding layer(s) of each model

        Args:
          inputs: a list of unicode strings, whitespace tokenized. Each list item i corresponds to the input for
            model_i. If a model uses >1 factor, tokens will still be joined by `factor_separator`
          factor_separator: a string used to separate a model's input factors

        Returns:
          TODO: confirm correct dimensionality
          mapped_inputs: list of np.arrays, each with dimensionality (factors, time, 1)


        """
        assert len(inputs) == len(self.fs_init), 'We need an input for each model'

        mapped_inputs = []
        for i, model_input in enumerate(inputs):
            # Nematus needs encoded utf-8 as input
            if type(model_input) is unicode:
                model_input = model_input.encode('utf8')
            tokens = model_input.strip().split()
            mapped_input = []
            for token in tokens:
                # if there's only one factor, allow the separator to occur in tokens
                if len(self.word_dicts[i]['input_dicts']) == 1:
                    token = [self.word_dicts[i]['input_dicts'][0].get(token, 1)]
                else:
                    token = [self.word_dicts[i]['input_dicts'][j].get(f, 1)
                             for j, f in enumerate(token.split(factor_separator))]

                mapped_input.append(token)

            # append the eos index
            mapped_input += [[0] * len(self.word_dicts[i]['input_dicts'])]
            mapped_inputs.append(numpy.array(mapped_input).T.reshape(len(mapped_input[0]), len(mapped_input), 1))

        return mapped_inputs

    # Note: this method could actually be fully implemented in the base class
    def map_constraints(self, constraint_token_seqs):
        """Map constraint sequences into the model's output vocabulary

        Args:
          constraint_token_seqs: a list of sequences of unicode strings corresponding to lexical constraints

        Returns:
          a list of sequences of ints corresponding to the constraint token indices in the output vocabulary

        """
        constraint_seqs = []
        for token_seq in constraint_token_seqs:
            if type(token_seq) is str:
                token_seq = token_seq.split()
            elif type(token_seq) is unicode:
                # Nematus needs encoded utf-8 as input
                token_seq = token_seq.encode('utf8').split()

            assert type(token_seq) is list or type(token_seq) is tuple, 'Constraint token seqs must be lists or tuples'
            # Note: all models share the same output dictionary, so we just use the first one
            token_idxs = [self.word_dicts[0]['output_dict'].get(token, 1) for token in token_seq]
            constraint_seqs.append(token_idxs)
        return constraint_seqs

    def start_hypothesis(self, inputs, constraints):
        """Compute the initial representation for each model, build the start hypothesis"""

        assert len(inputs) == self.num_models, 'Number of inputs must match the number of models'

        # Note: explicit initialization of coverage
        coverage = [numpy.zeros(l, dtype='int16')
                    for l in [len(s) for s in constraints]]

        next_states = [None] * self.num_models
        # contexts will be static throughout decoding
        contexts = [None] * self.num_models

        # BOS index
        next_w = -1 * numpy.ones((1,)).astype('int64')

        for i, model_input in enumerate(inputs):
            ret = self.fs_init[i](model_input)
            next_states[i] = numpy.tile(ret[0], (1,1))
            contexts[i] = ret[1]

        # the payload contains everything that the next timestep will need to generate another output
        payload = {
            'next_states': next_states,
            'contexts': contexts,
            'next_w': next_w
        }

        start_hyp = ConstraintHypothesis(
            token=None,
            score=None,
            coverage=coverage,
            constraints=constraints,
            payload=payload,
            backpointer=None,
            constraint_index=None,
            unfinished_constraint=False
        )

        return start_hyp

    def generate(self, hyp, n_best):
        """
        Generate the `n_best` hypotheses starting with `hyp`

        """

        # if we already generated EOS and there are no constraints (vanilla beam search),
        #   there's only one option -- just continue it and copy the current cost
        if hyp.token == self.eos_token and len(hyp.constraints) == 0:
            new_hyp = ConstraintHypothesis(
                token=self.eos_token,
                score=hyp.score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=hyp.payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False
            )
            return [new_hyp]
        # if there are constraints, and we generated eos, this hyp is dead
        elif hyp.token == self.eos_token and len(hyp.constraints) > 0:
            return []

        next_states = [None] * self.num_models
        next_p = [None] * self.num_models

        for i in xrange(self.num_models):
            # Note: batch size is implicitly = 1
            inps = [hyp.payload['next_w'], hyp.payload['contexts'][i], hyp.payload['next_states'][i]]
            ret = self.fs_next[i](*inps)
            next_p[i], next_w_tmp, next_states[i] = ret[0], ret[1], ret[2]

            #if return_alignment:
            #    dec_alphas[i] = ret[3]

            #if suppress_unk:
            #    next_p[i][:,1] = -numpy.inf

        # now compute the combined scores
        weighted_scores, probs = self.combine_model_scores(next_p)
        flat_scores = weighted_scores.flatten()

        n_best_idxs = numpy.argsort(flat_scores)[:n_best]
        n_best_scores = flat_scores[n_best_idxs]

        next_hyps = []
        # create a new hypothesis for each of the n-best
        for token_idx, score in zip(n_best_idxs, n_best_scores):
            if hyp.score is not None:
                next_score = hyp.score + score
            else:
                # hyp.score is None for the start hyp
                next_score = score

            payload = {
                'next_states': next_states,
                'contexts': hyp.payload['contexts'],
                'next_w': numpy.array([token_idx]).astype('int64')
            }

            new_hyp = ConstraintHypothesis(
                token=self.word_dicts[0]['output_idict'][token_idx],
                score=next_score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False
            )

            next_hyps.append(new_hyp)

        return next_hyps

    def generate_constrained(self, hyp):
        """
        Use hyp.constraints and hyp.coverage to return new hypothesis which start constraints
        that are not yet covered by this hypothesis.

        """
        assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'

        next_states = [None] * self.num_models
        next_p = [None] * self.num_models

        for i in xrange(self.num_models):
            # Note: batch size is implicitly = 1
            inps = [hyp.payload['next_w'], hyp.payload['contexts'][i], hyp.payload['next_states'][i]]
            ret = self.fs_next[i](*inps)
            next_p[i], next_w_tmp, next_states[i] = ret[0], ret[1], ret[2]

            #if return_alignment:
            #    dec_alphas[i] = ret[3]

            #if suppress_unk:
            #    next_p[i][:,1] = -numpy.inf

        # now compute the combined scores
        weighted_scores, probs = self.combine_model_scores(next_p)
        flat_scores = weighted_scores.flatten()

        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()
        for idx in available_constraints:
            constraint_idx = hyp.constraints[idx][0]
            constraint_score = flat_scores[constraint_idx]
            if hyp.score is not None:
                next_score = hyp.score + constraint_score
            else:
                # hyp.score is None for the start hyp
                next_score = constraint_score

            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1

            if len(coverage[idx]) > 1:
                unfinished_constraint = True
            else:
                unfinished_constraint = False

            payload = {
                'next_states': next_states,
                'contexts': hyp.payload['contexts'],
                'next_w': numpy.array([constraint_idx]).astype('int64')
            }

            new_hyp = ConstraintHypothesis(token=self.word_dicts[0]['output_idict'][constraint_idx],
                                           score=next_score,
                                           coverage=coverage,
                                           constraints=hyp.constraints,
                                           payload=payload,
                                           backpointer=hyp,
                                           constraint_index=(idx, 0),
                                           unfinished_constraint=unfinished_constraint)

            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps

    def continue_constrained(self, hyp):
        assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

        next_states = [None] * self.num_models
        next_p = [None] * self.num_models

        for i in xrange(self.num_models):
            # Note: batch size is implicitly = 1
            inps = [hyp.payload['next_w'], hyp.payload['contexts'][i], hyp.payload['next_states'][i]]
            ret = self.fs_next[i](*inps)
            next_p[i], next_w_tmp, next_states[i] = ret[0], ret[1], ret[2]

            #if return_alignment:
            #    dec_alphas[i] = ret[3]

            #if suppress_unk:
            #    next_p[i][:,1] = -numpy.inf

        # now compute the combined scores
        weighted_scores, probs = self.combine_model_scores(next_p)
        flat_scores = weighted_scores.flatten()

        constraint_row_index = hyp.constraint_index[0]
        # the index of the next token in the constraint
        constraint_tok_index = hyp.constraint_index[1] + 1
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

        # get the score for this token from the logprobs
        next_score = hyp.score + flat_scores[continued_constraint_token]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1

        if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
            unfinished_constraint = True
        else:
            unfinished_constraint = False

        payload = {
            'next_states': next_states,
            'contexts': hyp.payload['contexts'],
            'next_w': numpy.array([continued_constraint_token]).astype('int64')
        }

        new_hyp = ConstraintHypothesis(token=self.word_dicts[0]['output_idict'][continued_constraint_token],
                                       score=next_score,
                                       coverage=coverage,
                                       constraints=hyp.constraints,
                                       payload=payload,
                                       backpointer=hyp,
                                       constraint_index=constraint_index,
                                       unfinished_constraint=unfinished_constraint)

        return new_hyp

    def combine_model_scores(self, scores):
        """Use the weights to combine the scores from each model"""

        assert len(scores) == self.num_models, 'we need a vector of scores for each model in the ensemble'
        # this hack lets us do ad-hoc truncation of the vocabulary if we need to
        scores = [a[:, :self.word_dicts[i]['trg_size']-1] if self.word_dicts[i]['trg_size'] is not None else a
                  for i, a in enumerate(scores)]
        scores = numpy.array(scores)

        # Note: this is another implicit batch size = 1 assumption
        scores = numpy.squeeze(scores, axis=1)

        # Note the negative sign here, letting us treat the score as a cost to minimize
        weighted_scores = numpy.sum(-numpy.log(scores) * self.model_weights[:, numpy.newaxis], axis=0)

        # We dont use the model weights with probs because we want them to sum to 1
        probs = numpy.sum(scores, axis=0) / float(self.num_models)
        return weighted_scores, probs


