from collections import defaultdict, OrderedDict
import numpy as np
from sortedcontainers import SortedListWithKey


# Utility Functions
def init_coverage(constraints):
    coverage = []
    for c in constraints:
        coverage.append(np.zeros(len(c), dtype='int16'))
    return coverage


class ConstraintHypothesis:
    """A (partial) hypothesis which maintains an additional constraint coverage object

    Args:
        token (unicode): the surface form of this hypothesis
        score (float): the score of this hypothesis (higher is better)
        coverage (list of lists): a representation of the area of the constraints covered by this hypothesis
        constraints (list of lists): the constraints that may be used with this hypothesis
        payload (:obj:): additional data that comes with this hypothesis. Functions may
            require certain data to be present in the payload, such as the previous states, glimpses, etc...
        backpointer (:obj:`ConstraintHypothesis`): a pointer to the hypothesis object which generated this one
        constraint_index (tuple): if this hyp is part of a constraint, the index into `self.constraints` which
            is covered by this hyp `(constraint_idx, token_idx)`
        unfinished_constraint (bool): a flag which indicates whether this hyp is inside an unfinished constraint

    """

    def __init__(self, token, score, coverage, constraints, payload=None, backpointer=None,
                 constraint_index=None, unfinished_constraint=False):
        self.token = token
        self.score = score

        assert len(coverage) == len(constraints), 'constraints and coverage length must match'
        assert all(len(cov) == len(cons) for cov, cons in zip(coverage, constraints)), \
            'each coverage and constraint vector must match'

        self.coverage = coverage
        self.constraints = constraints
        self.backpointer = backpointer
        self.payload = payload
        self.constraint_index = constraint_index
        self.unfinished_constraint = unfinished_constraint

    def __str__(self):
        return u'token: {}, sequence: {}, score: {}, coverage: {}, constraints: {},'.format(
            self.token, self.sequence, self.score, self.coverage, self.constraints)

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def sequence(self):
        sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            sequence.append(current_hyp.token)
            current_hyp = current_hyp.backpointer
        sequence.append(current_hyp.token)
        return sequence[::-1]

    @property
    def constraint_index_sequence(self):
        constraint_sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            constraint_sequence.append(current_hyp.constraint_index)
            current_hyp = current_hyp.backpointer
        constraint_sequence.append(current_hyp.constraint_index)
        return constraint_sequence[::-1]

    def constraint_candidates(self):
        available_constraints = []
        for idx in range(len(self.coverage)):
            if self.coverage[idx][0] == 0:
                available_constraints.append(idx)

        return available_constraints


class AbstractBeam(object):

    def __init__(self, size):
        # note: here we assume bigger scores are better
        self.hypotheses = SortedListWithKey(key=lambda x: -x['score'])
        self.size = size

    def add(self, hyp):
        self.hypotheses.add(hyp)
        if len(self.hypotheses) > self.size:
            assert len(self.hypotheses) == self.size + 1
            del self.hypotheses[-1]

    def __len__(self):
        return len(self.hypotheses)

    def __iter__(self):
        for hyp in self.hypotheses:
            yield hyp


# FUNCTIONS USED BY THE CONSTRAINED DECODER
# Note: hyps on the top level may be finished (End with EOS), or may be continuing (haven't gotten an EOS yet)
class ConstrainedDecoder(object):

    def __init__(self, hyp_generation_func, constraint_generation_func, continue_constraint_func,
                 beam_implementation=AbstractBeam):
        self.hyp_generation_func = hyp_generation_func
        self.constraint_generation_func = constraint_generation_func
        self.continue_constraint_func = continue_constraint_func
        self.beam_implementation = beam_implementation

    # IMPLEMENTATION QUESTION: are mid-constraint hyps allowed to fall off of the beam or not?
    def search(self, start_hyp, constraints, max_source_len, beam_size=10):
        """create a constrained search
            - fill the search grid
        """

        # the total number of constraint tokens determines the height of the grid
        grid_height = sum(len(c) for c in constraints)

        search_grid = OrderedDict()

        # a beam with one hyp starts the search
        start_beam = self.beam_implementation(size=1)
        start_beam.add(start_hyp)

        search_grid[(0, 0)] = start_beam

        current_beam_count = 0
        for i in range(1, max_source_len + 1):
            print('TIME: {}'.format(i+1))
            j_start = max(i - (max_source_len - grid_height), 0)
            j_end = min(i, grid_height) + 1
            beams_in_i = j_end - j_start

            for j in range(j_start, min(i, grid_height) + 1):
                # create the new beam
                new_beam = self.beam_implementation(size=beam_size)
                # generate hyps from (i-1, j-1), and (i-1, j), and add them to the beam
                # cell to the left generates
                if (i-1, j) in search_grid:
                    generation_hyps = self.get_generation_hyps(search_grid[(i-1, j)])
                    for hyp in generation_hyps:
                        new_beam.add(hyp)
                # lower left diagonal cell adds hyps from constraints
                if (i-1, j-1) in search_grid:
                    new_constraint_hyps = self.get_new_constraint_hyps(search_grid[(i-1, j-1)])
                    continued_constraint_hyps = self.get_continued_constraint_hyps(search_grid[(i-1, j-1)])
                    for hyp in new_constraint_hyps:
                        new_beam.add(hyp)
                    for hyp in continued_constraint_hyps:
                        new_beam.add(hyp)

                search_grid[(i,j)] = new_beam
                print('index: {}'.format((i,j)))

            current_beam_count += beams_in_i
            assert len(search_grid) == current_beam_count, 'total grid size must be correct after adding new column'

        return search_grid

    def get_generation_hyps(self, beam):
        """return all hyps which are continuations of the hyps on this beam

        hyp_generation_func maps `(hyp) --> continuations`
          - the coverage vector of the parent hyp is not modified in each child
        """

        continuations = (self.hyp_generation_func(hyp) for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

    def get_new_constraint_hyps(self, beam):
        """return all hyps which start a new constraint from the hyps on this beam

        constraint_hyp_func maps `(hyp) --> continuations`
          - the coverage vector of the parent hyp is modified in each child
        """

        continuations = (self.constraint_generation_func(hyp)
                         for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

    def get_continued_constraint_hyps(self, beam):
        """return all hyps which continue the unfinished constraints on this beam

        constraint_hyp_func maps `(hyp, constraints) --> forced_continuations`
          - the coverage vector of the parent hyp is modified in each child
        """
        continuations = (self.continue_constraint_func(hyp)
                         for hyp in beam if hyp.unfinished_constraint)

        return continuations

