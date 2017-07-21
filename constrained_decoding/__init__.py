from collections import defaultdict, OrderedDict
import numpy as np
from sortedcontainers import SortedListWithKey


# Utility Functions
def init_coverage(constraints):
    coverage = []
    for c in constraints:
        coverage.append(np.zeros(len(c), dtype='int16'))
    return coverage


def create_constrained_decoder(translation_model):
    """
    Create a constrained decoder from a translation model that implements `translation_model.AbstractConstrainedTM`

    Args:
        translation_model (AbstractConstrainedTM): the translation model

    Returns:
        a new ConstrainedDecoder instance
    """
    decoder = ConstrainedDecoder(hyp_generation_func=translation_model.generate,
                                 constraint_generation_func=translation_model.generate_constrained,
                                 continue_constraint_func=translation_model.continue_constrained,
                                 beam_implementation=Beam)
    return decoder


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
        if type(token) == str:
            token = token.decode('utf8')
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
    def constraint_indices(self):
        """Return the (start, end) indexes of the constraints covered by this hypothesis"""

        # we know a hyp covered a constraint token if a coverage index changed from 0-->1
        # we also know which constraint and which token was covered by looking at the indices in the coverage
        # constraint tracker sequence = [None, (constraint_index, constraint_token_index), ...]
        # for each hyp:
        #     compare current_hyp.coverage with current_hyp.backpointer.coverage:
        #         if there is a difference:
        #             add difference to constraint tracker sequence
        #         else:
        #             add None to constraint tracker sequence

        def _compare_constraint_coverage(coverage_one, coverage_two):
            for constraint_idx, (this_coverage_row,
                                 prev_coverage_row) in enumerate(zip(coverage_one,
                                                                     coverage_two)):

                for token_idx, (this_coverage_bool,
                                prev_coverage_bool) in enumerate(zip(this_coverage_row, prev_coverage_row)):
                    if this_coverage_bool != prev_coverage_bool:
                        return constraint_idx, token_idx

            return None

        constraint_tracker_sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            # compare current_hyp.coverage with previous hyp's coverage
            constraint_tracker_sequence.append(_compare_constraint_coverage(current_hyp.coverage,
                                                                            current_hyp.backpointer.coverage))
            current_hyp = current_hyp.backpointer

        # we need to check if the first hyp covered a constraint
        start_coverage = init_coverage(current_hyp.constraints)
        constraint_tracker_sequence.append(_compare_constraint_coverage(current_hyp.coverage,
                                                                        start_coverage))

        # finally reverse this sequence to put it in order
        return constraint_tracker_sequence[::-1]

    @property
    def alignments(self):
        current_hyp = self
        if current_hyp.payload.get('alignments', None) is not None:
            alignment_weights = []
            while current_hyp.backpointer is not None:
                alignment_weights.append(current_hyp.payload['alignments'])
                current_hyp = current_hyp.backpointer
            return np.squeeze(np.array(alignment_weights[::-1]), axis=1)
        else:
            return None

    def constraint_candidates(self):
        available_constraints = []
        for idx in range(len(self.coverage)):
            if self.coverage[idx][0] == 0:
                available_constraints.append(idx)

        return available_constraints


# Beam Constraints (aka "Filters"): Functions which specify True/False checks that need to pass for a hyp to be added to the beam
# TODO: move this into the `filter` API
def unfinished(hyp, eos=u'</S>'):
    if hyp.token == eos:
        return False
    return True


def eos_covers_constraints(hyp, eos=set(['<eos>', u'</S>'])):
    """
    Return False if hyp.token is <eos>, and hyp does not cover all constraints, True otherwise

    """
    constraints_remaining = True
    coverage = hyp.coverage
    if sum(covered for cons in coverage for covered in cons) == sum(len(c) for c in coverage):
        constraints_remaining = False
    is_eos = False
    if hyp.token in eos:
        is_eos = True

    if constraints_remaining and is_eos:
        return False
    return True


class Beam(object):

    def __init__(self, size, lower_better=True):
        # are bigger scores better or worse?
        if lower_better:
            self.hypotheses = SortedListWithKey(key=lambda x: x['score'])
        else:
            self.hypotheses = SortedListWithKey(key=lambda x: -x['score'])

        self.size = size

    def add(self, hyp, beam_constraints=[]):
        if all(check(hyp) for check in beam_constraints):
            self.hypotheses.add(hyp)
            if len(self.hypotheses) > self.size:
                assert len(self.hypotheses) == self.size + 1
                del self.hypotheses[-1]

    def __len__(self):
        return len(self.hypotheses)

    def __iter__(self):
        for hyp in self.hypotheses:
            yield hyp


# Note: hyps on the top level may be finished (End with EOS), or may be continuing (haven't gotten an EOS yet)
# Note: because of the way we create new Beams, we would need to wrap the Beam class to access the `lower_better` kwarg
class ConstrainedDecoder(object):

    def __init__(self, hyp_generation_func, constraint_generation_func, continue_constraint_func,
                 beam_implementation=Beam):
        self.hyp_generation_func = hyp_generation_func
        self.constraint_generation_func = constraint_generation_func
        self.continue_constraint_func = continue_constraint_func
        self.beam_implementation = beam_implementation

        # TODO: allow user-specified beam_constraints as filters
        # TODO: allow additional args to filter functions
        # TODO: general factory DSL for filter functions
        self.beam_constraints = [eos_covers_constraints]

    # IMPLEMENTATION QUESTION: are mid-constraint hyps allowed to fall off of the beam or not?
    def search(self, start_hyp, constraints, max_hyp_len, beam_size=10):
        """Perform a constrained search
            - fill the search grid
        """

        # the total number of constraint tokens determines the height of the grid
        grid_height = sum(len(c) for c in constraints)

        search_grid = OrderedDict()

        # a beam with one hyp starts the search
        start_beam = self.beam_implementation(size=1)
        start_beam.add(start_hyp)

        search_grid[(0, 0)] = start_beam

        current_beam_count = 1
        for i in range(1, max_hyp_len + 1):
            j_start = max(i - (max_hyp_len - grid_height), 0)
            j_end = min(i, grid_height) + 1
            beams_in_i = j_end - j_start

            for j in range(j_start, min(i, grid_height) + 1):
                # create the new beam
                new_beam = self.beam_implementation(size=beam_size)
                # generate hyps from (i-1, j-1), and (i-1, j), and add them to the beam
                # cell to the left generates
                if (i-1, j) in search_grid:
                    generation_hyps = self.get_generation_hyps(search_grid[(i-1, j)], beam_size)
                    for hyp in generation_hyps:
                        new_beam.add(hyp, beam_constraints=self.beam_constraints)
                # lower left diagonal cell adds hyps from constraints
                if (i-1, j-1) in search_grid:
                    new_constraint_hyps = self.get_new_constraint_hyps(search_grid[(i-1, j-1)])
                    continued_constraint_hyps = self.get_continued_constraint_hyps(search_grid[(i-1, j-1)])
                    for hyp in new_constraint_hyps:
                        new_beam.add(hyp)
                    for hyp in continued_constraint_hyps:
                        new_beam.add(hyp)

                search_grid[(i,j)] = new_beam

            current_beam_count += beams_in_i

            # TODO: there is a bug where this assert can break, but why?
            #assert len(search_grid) == current_beam_count, 'total grid size must be correct after adding new column'

        return search_grid

    def get_generation_hyps(self, beam, beam_size=1):
        """return all hyps which are continuations of the hyps on this beam

        hyp_generation_func maps `(hyp) --> continuations`
          - the coverage vector of the parent hyp is not modified in each child
        """

        continuations = (self.hyp_generation_func(hyp, beam_size) for hyp in beam if not hyp.unfinished_constraint)

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

    @staticmethod
    def best_n(search_grid, eos_token, n_best=1, cut_off_eos=True, return_model_scores=False, return_alignments=False,
               length_normalization=True):
        top_row = max(k[1] for k in search_grid.keys())

        if top_row > 1:
            output_beams = [search_grid[k] for k in search_grid.keys() if k[1] == top_row]
        else:
            # constraints seq is empty
            # Note this is a very hackish way to get the last beam
            output_beams = [search_grid[search_grid.keys()[-1]]]

        output_hyps = [h for beam in output_beams for h in beam]

        # getting the true length of each hypothesis
        true_lens = [h.sequence.index(eos_token) if eos_token in h.sequence else len(h.sequence)
                     for h in output_hyps]
        true_lens = [float(l) for l in true_lens]
        # hack to let us keep true_len info after sorting
        for h, true_len in zip(output_hyps, true_lens):
            h.true_len = true_len

        # if at least one hyp ends with eos, drop all the ones that don't (note this makes some big assumptions)
        #eos_hyps = [h for h in output_hyps if eos_token in h.sequence]
        #if len(eos_hyps) > 0:
        #    output_hyps = eos_hyps

        # normalizing scores by true_len is optional -- Note: length norm param could also be weighted as in GNMT paper
        try:
            if length_normalization:
                output_seqs = [(h.sequence, h.score / true_len, h) for h, true_len in zip(output_hyps, true_lens)]
            else:
                output_seqs = [(h.sequence, h.score, h) for h in output_hyps]
        except:
            # Note: this happens when there is actually no output, just a None
            output_seqs = [([eos_token], float('inf'), None)]

        if cut_off_eos:
            output_seqs = [(seq[:int(t_len)], score, h) for (seq, score, h), t_len in zip(output_seqs, true_lens)]

        # sort by score, lower is better (i.e. cost semantics)
        output_seqs = sorted(output_seqs, key=lambda x: x[1])
        if return_alignments:
            assert output_hyps[0].alignments is not None, 'Cannot return alignments if they are not part of hypothesis payloads'
            # we subtract 1 from true len index because the starting `None` token is not included in the `h.alignments`
            alignments = [h.alignments[:int(h.true_len-1)] for seq, score, h in output_seqs]

        if return_model_scores:
            output_seqs = [(seq, score, h.payload['model_scores'] / true_len)
                           for (seq, score, h), true_len in zip(output_seqs, true_lens)]
        else:
            output_seqs = [(seq, score, h) for seq, score, h in output_seqs]

        if return_alignments:
            if n_best > 1:
                return output_seqs[:n_best], alignments[:n_best]
            else:
                return output_seqs[0], alignments[:1]
        else:
            if n_best > 1:
                return output_seqs[:n_best]
            else:
                # Note in this case we don't return a list
                return output_seqs[0]


