import unittest
import copy

import numpy as np

from constrained_decoding.translation_model import AbstractConstrainedTM
from constrained_decoding import ConstraintHypothesis, init_coverage


# DUMBEST POSSIBLE IMPLEMENTATION of generation functions
# Note that generation and search are done by _different_ classes
class DumbTranslationModel(AbstractConstrainedTM):

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def start_hypothesis(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, hyp, n_best=1):
        # make k_best random hyp objects
        next_tokens = np.random.choice(self.vocabulary, size=n_best)
        next_scores = np.random.random(size=n_best)

        new_hyps = []
        for i in range(n_best):
            new_hyp = ConstraintHypothesis(token=next_tokens[i],
                                           score=next_scores[i],
                                           coverage=copy.deepcopy(hyp.coverage),
                                           constraints=hyp.constraints,
                                           payload=None,
                                           backpointer=hyp,
                                           constraint_index=None,
                                           unfinished_constraint=False
                                          )
            new_hyps.append(new_hyp)

        return new_hyps

    def generate_constrained(self, hyp):
        """Look at the coverage of the hyp to get constraint candidates"""

        assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'
        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()
        for idx in available_constraints:
            # starting a new constraint
            constraint_token = hyp.constraints[idx][0]
            # this should come from the model
            score = np.random.random()
            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1
            if len(coverage[idx]) > 1:
                unfinished_constraint = True
            else:
                unfinished_constraint = False

            new_hyp = ConstraintHypothesis(token=constraint_token,
                                           score=score,
                                           coverage=coverage,
                                           constraints=hyp.constraints,
                                           payload=None,
                                           backpointer=hyp,
                                           constraint_index=(idx, 0),
                                           unfinished_constraint=unfinished_constraint
                                          )
            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps

    def continue_constrained(self, hyp):
        assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

        # this should come from the model
        score = np.random.random()

        constraint_row_index = hyp.constraint_index[0]
        # the index of the next token in the constraint
        constraint_tok_index = hyp.constraint_index[1] + 1
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1

        if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
            unfinished_constraint = True
        else:
            unfinished_constraint = False

        new_hyp = ConstraintHypothesis(token=continued_constraint_token,
                                       score=score,
                                       coverage=coverage,
                                       constraints=hyp.constraints,
                                       payload=None,
                                       backpointer=hyp,
                                       constraint_index=constraint_index,
                                       unfinished_constraint=unfinished_constraint)

        return new_hyp


class TestConstrainedHypothesis(unittest.TestCase):

    def setUp(self):

        start_symbol = 0

        sample_constraints = [
            [1, 2],
            [5, 6, 7]
        ]
        coverage = [
            [0, 0],
            [0, 0, 0]
        ]

        p_start = 1.0

        # dummy start hypothesis
        self.start_hyp = ConstraintHypothesis(token=start_symbol,
                                              score=p_start,
                                              coverage=coverage,
                                              constraints=sample_constraints,
                                              constraint_index=None,
                                              payload=None,
                                              backpointer=None,
                                              unfinished_constraint=False)

    def test_chaining_hypotheses(self):
        next_symbol = 1
        next_hyp = ConstraintHypothesis(token=next_symbol,
                                        score=1.0,
                                        coverage=self.start_hyp.coverage,
                                        constraints=self.start_hyp.constraints,
                                        constraint_index=None,
                                        payload=None,
                                        backpointer=self.start_hyp,
                                        unfinished_constraint=False)

        # test that hypotheses correctly output their sequences
        token_seq = next_hyp.sequence
        true_seq = [0, 1]
        self.assertTrue(len(token_seq) == 2)
        self.assertTrue(all(true_sym == seq_sym
                            for true_sym, seq_sym in zip(true_seq, token_seq)))

        constraint_idx_seq = next_hyp.constraint_index_sequence
        true_seq = [None, None]
        self.assertTrue(len(constraint_idx_seq) == 2)
        self.assertTrue(all(true_sym == seq_sym
                            for true_sym, seq_sym in zip(true_seq, constraint_idx_seq)))

    def test_constraint_indices(self):
        next_symbol = 1
        next_coverage = [
            [1, 0],
            [0, 0, 0]
        ]
        constraint_index = (0, 0)
        next_hyp = ConstraintHypothesis(token=next_symbol,
                                        score=1.0,
                                        coverage=next_coverage,
                                        constraints=self.start_hyp.constraints,
                                        constraint_index=constraint_index,
                                        payload=None,
                                        backpointer=self.start_hyp,
                                        unfinished_constraint=False)

        # test that hypotheses correctly output their sequences
        constraint_idx_seq = next_hyp.constraint_index_sequence
        true_seq = [None, constraint_index]
        self.assertTrue(len(constraint_idx_seq) == 2)
        self.assertTrue(all(true_sym == seq_sym
                            for true_sym, seq_sym in zip(true_seq, constraint_idx_seq)))


class TestConstrainedDecoder(unittest.TestCase):

    def setUp(self):
        pass


class TestTranslationModel(unittest.TestCase):

    def setUp(self):
        self.vocabulary = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        sample_constraints = [
            [1, 2],
            [5, 6, 7]
        ]

        # create a start hypothesis
        start_token = u'<S>'
        p_start = 1.0

        # dummy start hypothesis
        self.start_hyp = ConstraintHypothesis(token=start_token,
                                              score=p_start,
                                              coverage=init_coverage(sample_constraints),
                                              constraints=sample_constraints,
                                              payload=None,
                                              backpointer=None,
                                              unfinished_constraint=False)

    def test_translation_model(self):
        dumb_tm = DumbTranslationModel(vocabulary=self.vocabulary)
        t1_hyps = dumb_tm.generate(self.start_hyp, n_best=1)
        self.assertEqual(len(t1_hyps), 1)
        t1_hyps = dumb_tm.generate(self.start_hyp, n_best=5)
        self.assertEqual(len(t1_hyps), 5)










