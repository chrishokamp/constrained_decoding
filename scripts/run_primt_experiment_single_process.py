"""
Run a Pick-Revise experiment for N cycles, evaluate each file for each cycle

Params
------
Num Cycles
Source File
Reference File
Constraint selection method
Saved Model to use

"""
from __future__ import print_function

import logging
import argparse
import codecs
import itertools
import errno
import json
import shutil
import time
import re
import os
from collections import Counter
from multiprocessing import Process, Queue
from subprocess import Popen, PIPE

import numpy as np

from constrained_decoding.translation_model.nmt import NeuralTranslationModel
from constrained_decoding import ConstrainedDecoder, Beam

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: move this to arg
BLEU_SCRIPT = '/home/chokamp/projects/neural_mt/test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl'

def compute_bleu_score(hyp_file, ref_file):
    multibleu_cmd = ['perl', BLEU_SCRIPT, ref_file, '<']
    mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)
    with codecs.open(hyp_file, encoding='utf8') as hyps:
        for l in hyps.read().strip().split('\n'):
            # send the line to the BLEU script
            print(l.encode('utf8'), file=mb_subprocess.stdin)
            mb_subprocess.stdin.flush()

    # send end of file, read output.
    mb_subprocess.stdin.close()
    stdout = mb_subprocess.stdout.readline()
    logger.info(stdout)
    out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
    assert out_parse is not None

    # extract the score
    bleu_score = float(out_parse.group()[6:])
    logger.info('BLEU SCORE: {}'.format(bleu_score))
    mb_subprocess.terminate()
    return bleu_score


# WORKING: add different ways of getting constraints from the reference
def get_max_ref_start_constraint(hyp, ref, max_constraint_cutoff=3, ref_vocab=None):
    ref_constraints = []
    hyp_toks = set(hyp)

    # get constraints bounded by unknown toks on either side
    current_sub_seq = []
    sub_seq_counter = 0
    for tok in ref:
        if not tok in hyp_toks or (sub_seq_counter > 0 and sub_seq_counter <= max_constraint_cutoff):
            current_sub_seq.append(tok)
            sub_seq_counter += 1
        else:
            if len(current_sub_seq) > 0:
                ref_constraints.append(current_sub_seq)
                current_sub_seq = []
                sub_seq_counter = 0
    if len(current_sub_seq) > 0:
        ref_constraints.append(current_sub_seq)

    longest_constraint_idx = 0
    len_longest = 0
    for c_i, c in enumerate(ref_constraints):
        if len(c) > len_longest:
            len_longest = len(c)
            longest_constraint_idx = c_i

    if len(ref_constraints) > 0:
        longest_constraint = ref_constraints[longest_constraint_idx][:max_constraint_cutoff]
    else:
        longest_constraint = []

    # Sanity: assert that the model knows about this constraint
    if ref_vocab is not None:
        for constraint_i in ref_constraints:
            try:
                assert all(w in ref_vocab for w in constraint_i)
            except:
                raise ValueError('if a constraint is not in the index, theres a problem')

    return (ref_constraints, longest_constraint)


def get_max_ref_constraint(hyp, ref, max_constraint_cutoff=3, ref_vocab=None):
    ref_constraints = []
    hyp_toks = set(hyp)

    current_sub_seq = []
    for tok in ref:
        if not tok in hyp_toks:
            current_sub_seq.append(tok)
        else:
            if len(current_sub_seq) > 0:
                ref_constraints.append(current_sub_seq)
                current_sub_seq = []
    if len(current_sub_seq) > 0:
        ref_constraints.append(current_sub_seq)

    longest_constraint_idx = 0
    len_longest = 0
    for c_i, c in enumerate(ref_constraints):
        if len(c) > len_longest:
            len_longest = len(c)
            longest_constraint_idx = c_i

    if len(ref_constraints) > 0:
        longest_constraint = ref_constraints[longest_constraint_idx][:max_constraint_cutoff]
    else:
        longest_constraint = []

    # Sanity: assert that the model knows about this constraint
    if ref_vocab is not None:
        for constraint_i in ref_constraints:
            try:
                assert all(w in ref_vocab for w in constraint_i)
            except:
                raise ValueError('if a constraint is not in the index, theres a problem')

    return (ref_constraints, longest_constraint)


def create_constraints(hyp_file, ref_file, constraint_gen_func, max_constraint_cutoff=3, ref_vocab=None):
    with codecs.open(hyp_file, encoding='utf8') as hyp_input:
        with codecs.open(ref_file, encoding='utf8') as ref_input:
            hyps = [l.split() for l in hyp_input.read().strip().split('\n')]
            refs = [l.split() for l in ref_input.read().strip().split('\n')]
    assert len(hyps) == len(refs), u'We need the same number of hyps and refs'

    constraint_lists, longest_constraints = zip(*[constraint_gen_func(hyp, ref, max_constraint_cutoff,
                                                                      ref_vocab=ref_vocab)
                                                  for hyp, ref in zip(hyps, refs)])
    return longest_constraints


def create_constrained_decoder(translation_model):
    decoder = ConstrainedDecoder(hyp_generation_func=translation_model.generate,
                                 constraint_generation_func=translation_model.generate_constrained,
                                 continue_constraint_func=translation_model.continue_constrained,
                                 beam_implementation=Beam)
    return decoder


def create_translation_model(config_file):
    ntm = NeuralTranslationModel(config_file=config_file)
    return ntm


def decode_input(ntm, decoder, source, constraints, length_factor=1.3):
    # TODO: this is a hack until we remove the target_prefix completely from the graph
    target_prefix = u'<S>'.split()
    target_prefix_ = ntm.imt_model.map_idx_or_unk(target_prefix,
                                                  ntm.imt_model.trg_vocab,
                                                  ntm.imt_model.unk_idx)

    # Note: tile 1x because minibatch size is effectively 1
    target_prefix_ = np.tile(target_prefix_, (1, 1))

    source_, constraints_ = ntm.build_input_representations(source, constraints)
    start_hyp = ntm.start_hypothesis(source_seq=source_, target_prefix=target_prefix_,
                                     constraints=constraints_)
    search_grid = decoder.search(start_hyp=start_hyp, constraints=constraints_,
                                 max_hyp_len=int(round(len(source) * length_factor)),
                                 beam_size=5)
    best_output = decoder.best_n(search_grid, ntm.eos_token, n_best=1)
    return best_output


def run_primt_cycle(source_file, target_file, config_file, output_dir, cycle_idx, constraints):
    
    # TODO: only create these once, not when this func is called
    ntm = create_translation_model(config_file)
    decoder = create_constrained_decoder(ntm)

    # input and output queues used by the processes
    source_seqs = []
    target_seqs = []
    with codecs.open(source_file, encoding='utf8') as source_inp:
        with codecs.open(target_file, encoding='utf8') as target_inp:
            for line_idx, (src_l, trg_l, cons) \
                    in enumerate(itertools.izip(source_inp, target_inp, constraints)):
                src = src_l.strip().split()
                trg = trg_l.strip().split()

                source_seqs.append(src)
                target_seqs.append(trg)


    logger.info('Translating file: {}'.format(source_file))

    mkdir_p(output_dir)
    output_file_name = os.path.join(output_dir, 'primt.translations.{}'.format(cycle_idx))
    if os.path.exists(output_file_name):
        return output_file_name, ntm
        # overwrite older version if it exists
        # open(output_file_name, 'w')

    # Note: we also write constraints so the experiment can be redone without decoding again
    output_translations = []
    for i, (source, constraint) in enumerate(zip(source_seqs, constraints)):
        trans, score = decode_input(ntm, decoder, source, constraint)
        # hides trimming of the 'None' at the beginning of each translation within search or decoder
        trans = trans[1:]

        # Note hardcoded </S> for cutoff
        if u'</S>' in trans:
            trimmed_trans = trans[:trans.index(u'</S>')]
        else:
            trimmed_trans = trans
        output_translations.append((trans, score))
        with codecs.open(output_file_name, 'a', encoding='utf8') as out:
            out.write(u' '.join(trimmed_trans)  + u'\n')

        if i % 50 == 0:
            logger.info('Wrote {} lines to {}'.format(i+1, output_file_name))

    return output_file_name, ntm


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",
                        help="the source text for the test corpus")
    parser.add_argument("-t", "--target",
                        help="the target text for the test corpus")
    parser.add_argument("-c", "--config", help="the yaml config file for the IMT system")
    parser.add_argument("-o", "--outputdir",
                        help="the directory where we should write the output files and the experiment report")
    parser.add_argument("-nc", "--numcycles", type=int,
                        help="the number of PRIMT cycles to perform")
    parser.add_argument("--strict_constraints", type=bool, default=True,
                        help="whether all reference words must be missing to build a new constraint")

    args = parser.parse_args()
    arg_dict = vars(args)

    score_file = os.path.join(args.outputdir, 'iteration_scores.BLEU')

    # overwrite old version if it exists
    if os.path.exists(score_file):
        logger.warn('{} already exists, moving to: {}'.format(score_file, score_file + '.old'))
        shutil.copyfile(score_file, score_file + '.old')
        open(score_file, 'w')

    if args.strict_constraints:
        constraint_gen_func = get_max_ref_constraint
    else:
        constraint_gen_func = get_max_ref_start_constraint
    logger.info('Strict generation of constraints is: {}'.format(args.strict_constraints))

    # start with nothing
    all_cycle_constraints = [[] for _ in codecs.open(args.source).read().strip().split('\n')]

    for cycle_idx in range(args.numcycles):
        logger.info('Running PRIMT cycle: {}'.format(cycle_idx))
        primt_output_file, tm = run_primt_cycle(args.source, args.target, args.config, args.outputdir, cycle_idx,
                                                all_cycle_constraints)

        cycle_constraints = create_constraints(primt_output_file, args.target, constraint_gen_func=constraint_gen_func,
                                               ref_vocab=tm.imt_model.trg_vocab)

        # add these constraints for the next cycle
        for cons_i, cons in enumerate(cycle_constraints):
            # Note if we add empty constraints decoding will break
            if len(cons) > 0:
                all_cycle_constraints[cons_i].append(cons)

        cycle_bleu = compute_bleu_score(primt_output_file, args.target)
        logger.info("CYCLE: {} BLEU: {}".format(cycle_idx, cycle_bleu))
        with codecs.open(score_file, 'a', encoding='utf8') as scores:
            scores.write("CYCLE: {} BLEU: {}\n".format(cycle_idx, cycle_bleu))

        with codecs.open(os.path.join(args.outputdir, 'cycle_constraints.{}.json'.format(cycle_idx)),
                         'w', encoding='utf8') as cons_out:
            cons_out.write(json.dumps(all_cycle_constraints, indent=2))

    logger.info('Finished translating {0} with constraints'.format(args.source))


