import copy
import numpy as np
from collections import defaultdict, OrderedDict

from . import AbstractConstrainedTM
from .. import ConstraintHypothesis

from nn_imt import IMTPredictor
from neural_mt.machine_translation.configurations import get_config


class NeuralTranslationModel(AbstractConstrainedTM):

    def __init__(self, config_file):
        """
        Intitialize the model according to user provided configuration. This Constrained Translation model uses the
        Interactive Neural Machine Translation model interface.

        - follow the style of BeamSearch, but remove the search logic
        - build the graph and load the parameters (i.e. create a Predictor and expose the right functions)
        """

        self.imt_model = IMTPredictor(get_config(config_file))
        self.imt_beam_search = self.imt_model.beam_search
        self.eos_token = u'</S>'

    def build_input_representations(self, source_tokens, constraint_token_seqs):
        """Encode the input sequences using the source and target word-->idx maps"""

        source_seq = self.imt_model.map_idx_or_unk(source_tokens,
                                                   self.imt_model.src_vocab,
                                                   self.imt_model.unk_idx)

        # Note: we assume that constraints are in the target language
        constraint_seqs = []
        for token_seq in constraint_token_seqs:
            token_idxs = self.imt_model.map_idx_or_unk(token_seq,
                                                       self.imt_model.trg_vocab,
                                                       self.imt_model.unk_idx)
            constraint_seqs.append(token_idxs)

        source_seq = np.tile(source_seq, (1, 1))

        # TODO: we'll need to tile constraint_seqs up to beam_size for NMT models that take constraints as inputs
        #input_ = numpy.tile(seq, (self.exp_config['beam_size'], 1))

        return (source_seq, constraint_seqs)

    # TODO: remove target_prefix from args (see below)
    # TODO: standardize this interface to only take (inputs, constraints)
    # TODO: this will let us keep the GBS implementation as abstract as possible
    # TODO: add `map_inputs` to `AbstractConstrainedTM` interface
    # TODO: add `map_constraints` to `AbstractConstrainedTM` interface -- split out constraint
    # TODO: mapping logic from `build_input_representations` above
    def start_hypothesis(self, source_seq, target_prefix, constraints, coverage=None):
        """
        Build the start hyp for a neural translation model.
        Models may or may not take constraints as inputs. I.e. by modeling
        the probability of generating vs. copying from the constraints.

        """

        # Note: there should actually be no self.target_sampling_input
        # Note: because we don't use the prefix representation in constrained decoding
        input_values = {
            self.imt_model.source_sampling_input: source_seq,
            self.imt_model.target_sampling_input: target_prefix
        }

        # Note that the initial input of an NMT model is currently implicit (i.e. Readout.initial_input)
        contexts, states, beam_size = self.imt_beam_search.compute_initial_states_and_contexts(inputs=input_values)

        # Note: explicit initialization of coverage
        coverage = [np.zeros(l, dtype='int16') for l in [len(s) for s in constraints]]

        # the payload contains everything that the next timestep will need to generate another output
        payload = {
            'contexts': contexts,
            'states': states,
            # input_values is here because of a bug in getting beam-size from the graph
            'input_values': input_values
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
        Note: the `n_best` parameter here is only used to limit the number of hypothesis objects that are generated
        from the input hyp, the beam implementation may specify a different `n_best`

        """

        # if we already generated EOS, there's only one option -- just continue it and copy the cost
        if hyp.token == self.eos_token:
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

        logprobs = self.imt_beam_search.compute_logprobs(hyp.payload['input_values'],
                                                         hyp.payload['contexts'],
                                                         hyp.payload['states'])

        assert len(logprobs) == 1, 'NMT logprob logic depends upon logprobs only having one row'
        n_best_outputs = np.argsort(logprobs.flatten())[:n_best]
        chosen_costs = logprobs.flatten()[n_best_outputs]

        # generate n_best ConstrainedHypothesis for each item on the beam, return them all
        # argsort logprobs
        payload = hyp.payload

        # Note: it's critical to use the OrderedDict here, otherwise args will get out of order in theano funcs
        tiled_payload = defaultdict(OrderedDict)
        tiled_payload['contexts']['attended'] = np.tile(payload['contexts']['attended'], (1, n_best, 1))
        tiled_payload['contexts']['attended_mask'] = np.tile(payload['contexts']['attended_mask'], (1, n_best))

        tiled_payload['states']['outputs'] = np.tile(payload['states']['outputs'], n_best)
        tiled_payload['states']['states'] = np.tile(payload['states']['states'], (n_best, 1))
        tiled_payload['states']['weights'] = np.tile(payload['states']['weights'], (n_best, 1))
        tiled_payload['states']['weighted_averages'] = np.tile(payload['states']['weighted_averages'], (n_best, 1))

        tiled_payload['input_values'][self.imt_model.source_sampling_input] = np.tile(payload['input_values'][self.imt_model.source_sampling_input],
                                                                                      (n_best, 1))
        tiled_payload['input_values'][self.imt_model.target_sampling_input] = np.tile(payload['input_values'][self.imt_model.target_sampling_input],
                                                                                      (n_best, 1))

        # Now we need to tile the previous hyp values to make this work
        next_states = self.imt_beam_search.compute_next_states(tiled_payload['input_values'],
                                                               tiled_payload['contexts'],
                                                               tiled_payload['states'],
                                                               n_best_outputs)

        # create ContstrainedHypothesis objects from these states (tile back down to one)
        new_hyps = []
        for hyp_idx in range(n_best):
            new_payload = defaultdict(OrderedDict)
            new_payload['contexts'] = payload['contexts']

            new_payload['states']['outputs'] = np.atleast_1d(next_states['outputs'][hyp_idx])
            new_payload['states']['states'] = np.atleast_2d(next_states['states'][hyp_idx])
            new_payload['states']['weights'] = np.atleast_2d(next_states['weights'][hyp_idx])
            new_payload['states']['weighted_averages'] = np.atleast_2d(next_states['weighted_averages'][hyp_idx])

            new_payload['input_values'] = hyp.payload['input_values']

            # TODO: account for EOS continuations -- i.e. make other costs infinite
            if hyp.score is not None:
                next_score = hyp.score + chosen_costs[hyp_idx]
            else:
                # hyp.score is None for the start hyp
                next_score = chosen_costs[hyp_idx]

            new_hyp = ConstraintHypothesis(
                token=self.imt_model.trg_ivocab[n_best_outputs[hyp_idx]],
                score=next_score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=new_payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False
            )

            new_hyps.append(new_hyp)

        return new_hyps

    def generate_constrained(self, hyp):
        """Use hyp.constraints and hyp.coverage to return new hypothesis which start constraints
        that are not yet covered by this hypothesis.

        """
        assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'

        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()

        # TODO: if the model knows about constraints, getting the score from the model must be done differently
        # TODO: currently, according to the model, there is no difference between generating and choosing from constraints
        logprobs = self.imt_beam_search.compute_logprobs(hyp.payload['input_values'],
                                                         hyp.payload['contexts'],
                                                         hyp.payload['states']).flatten()
        for idx in available_constraints:
            # start new constraints
            constraint_idx = hyp.constraints[idx][0]

            next_states = self.imt_beam_search.compute_next_states(hyp.payload['input_values'],
                                                                   hyp.payload['contexts'],
                                                                   hyp.payload['states'],
                                                                   np.atleast_1d(constraint_idx))

            new_payload = defaultdict(OrderedDict)
            new_payload['contexts'] = hyp.payload['contexts']

            new_payload['states'] = next_states

            new_payload['input_values'] = hyp.payload['input_values']

            # get the score for this token from the logprobs
            if hyp.score is not None:
                next_score = hyp.score + logprobs[constraint_idx]
            else:
                # hyp.score is None for the start hyp
                next_score = logprobs[constraint_idx]

            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1

            if len(coverage[idx]) > 1:
                unfinished_constraint = True
            else:
                unfinished_constraint = False

            # TODO: if the model knows about constraints, getting the score from the model must be done differently
            new_hyp = ConstraintHypothesis(token=self.imt_model.trg_ivocab[constraint_idx],
                                           score=next_score,
                                           coverage=coverage,
                                           constraints=hyp.constraints,
                                           payload=new_payload,
                                           backpointer=hyp,
                                           constraint_index=(idx, 0),
                                           unfinished_constraint=unfinished_constraint
                                          )
            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps

    def continue_constrained(self, hyp):
        assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

        # Note: if the model knows about constraints, getting the score from the model must be done differently
        # Note: according to this model, there is no difference between generating and choosing from constraints
        logprobs = self.imt_beam_search.compute_logprobs(hyp.payload['input_values'],
                                                         hyp.payload['contexts'],
                                                         hyp.payload['states']).flatten()

        constraint_row_index = hyp.constraint_index[0]
        # the index of the next token in the constraint
        constraint_tok_index = hyp.constraint_index[1] + 1
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

        # get the score for this token from the logprobs
        if hyp.score is not None:
            next_score = hyp.score + logprobs[continued_constraint_token]
        else:
            # hyp.score is None for the start hyp
            next_score = logprobs[continued_constraint_token]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1

        if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
            unfinished_constraint = True
        else:
            unfinished_constraint = False

        next_states = self.imt_beam_search.compute_next_states(hyp.payload['input_values'],
                                                               hyp.payload['contexts'],
                                                               hyp.payload['states'],
                                                               np.atleast_1d(continued_constraint_token))

        new_payload = defaultdict(OrderedDict)
        new_payload['contexts'] = hyp.payload['contexts']

        new_payload['states'] = next_states

        new_payload['input_values'] = hyp.payload['input_values']

        new_hyp = ConstraintHypothesis(token=self.imt_model.trg_ivocab[continued_constraint_token],
                                       score=next_score,
                                       coverage=coverage,
                                       constraints=hyp.constraints,
                                       payload=new_payload,
                                       backpointer=hyp,
                                       constraint_index=constraint_index,
                                       unfinished_constraint=unfinished_constraint)

        return new_hyp
