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


BLEU_SCRIPT = '/home/chris/projects/neural_mt/test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl'

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


def get_max_ref_constraint(hyp, ref, max_constraint_cutoff=3):
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

    return (ref_constraints, longest_constraint)


def create_constraints(hyp_file, ref_file, max_constraint_cutoff=3):
    with codecs.open(hyp_file, encoding='utf8') as hyp_input:
        with codecs.open(ref_file, encoding='utf8') as ref_input:
            hyps = [l.split() for l in hyp_input.read().strip().split('\n')]
            refs = [l.split() for l in ref_input.read().strip().split('\n')]
    assert len(hyps) == len(refs), u'We need the same number of hyps and refs'

    constraint_lists, longest_constraints = zip(*[get_max_ref_constraint(hyp, ref, max_constraint_cutoff)
                                                  for hyp, ref in zip(hyps, refs)])
    return longest_constraints


def create_constrained_decoder(translation_model):
    decoder = ConstrainedDecoder(hyp_generation_func=translation_model.generate,
                                 constraint_generation_func=translation_model.generate_constrained,
                                 continue_constraint_func=translation_model.continue_constrained,
                                 beam_implementation=Beam)
    return decoder


def translate_with_model(config_file, pid, input_queue, output_queue, length_factor=1.5, verbose=True):
    """Creates a model, uses the queues to communicate with the world"""
    ntm = NeuralTranslationModel(config_file=config_file)
    decoder = create_constrained_decoder(ntm)

    # TODO: this is a hack until we remove the target_prefix completely from the graph
    target_prefix = u'<S>'.split()
    target_prefix_ = ntm.imt_model.map_idx_or_unk(target_prefix,
                                                  ntm.imt_model.trg_vocab,
                                                  ntm.imt_model.unk_idx)

    # Note: tile 1x because minibatch size is effectively 1
    target_prefix_ = np.tile(target_prefix_, (1, 1))

    while True:
        req = input_queue.get()
        if req is None:
            break

        idx, source, constraints = req[0], req[1], req[2]
        if verbose:
            logger.info('{0} - Translating: {1}\n'.format(pid,idx))

        source_, constraints_ = ntm.build_input_representations(source, constraints)
        start_hyp = ntm.start_hypothesis(source_seq=source_, target_prefix=target_prefix_,
                                         constraints=constraints_)
        search_grid = decoder.search(start_hyp=start_hyp, constraints=constraints_,
                                     max_hyp_len=int(round(len(source) * length_factor)),
                                     beam_size=5)
        best_output = decoder.best_n(search_grid, ntm.eos_token, n_best=1)

        output_queue.put((idx, best_output))


# def translate_with_model(config_file, pid, input_queue, output_queue, length_factor=1.5, verbose=True):
def run_primt_cycle(source_file, target_file, config_file, output_dir, cycle_idx, constraints, num_parallel=1):

    # input and output queues used by the processes
    input_queue = Queue()
    output_queue = Queue()
    processes = [None for  _ in range(num_parallel)]
    for p_idx in xrange(num_parallel):
        # create the model
        processes[p_idx] = Process(target=translate_with_model,
                                   args=(config_file, p_idx, input_queue, output_queue, 1.5, True))
        processes[p_idx].start()

    logger.info('Created {} processes for translation'.format(num_parallel))

    def _send_jobs(src_file, trg_file, target_constraints):
        source_seqs = []
        target_seqs = []
        with codecs.open(src_file, encoding='utf8') as source_inp:
            with codecs.open(trg_file, encoding='utf8') as target_inp:
                for line_idx, (src_l, trg_l, cons) \
                        in enumerate(itertools.izip(source_inp, target_inp, target_constraints)):
                    src = src_l.split()
                    trg = trg_l.split()

                    source_seqs.append(src)
                    target_seqs.append(trg)

                    input_queue.put((line_idx, src, cons))

        return line_idx + 1, source_seqs, target_seqs

    def _finish_processes():
        for _ in xrange(num_parallel):
            input_queue.put(None)

    def _retrieve_jobs(num_sequences):
        outputs = [None for _ in xrange(num_sequences)]
        output_idx = 0
        for loop_idx in xrange(num_sequences):
            line_idx, translation = output_queue.get()
            outputs[line_idx] = translation
            if loop_idx % 100 == 0:
                logger.info('Translated {} / {}'.format(loop_idx+1, num_sequences))
            # yield as translations are ready in their original order
            while output_idx < num_sequences and outputs[output_idx] is not None:
                yield outputs[output_idx]
                output_idx += 1

    logger.info('Translating file: {}'.format(source_file))

    num_seqs_to_translate, source_sens, target_sens = _send_jobs(source_file, target_file, constraints)
    _finish_processes()

    mkdir_p(output_dir)
    output_file_name = os.path.join(output_dir, 'primt.translations.{}'.format(cycle_idx))
    # overwrite older version
    open(output_file_name, 'w')

    # TODO: also write constraints so the experiment can be redone without decoding again
    output_translations = []
    for i, (trans, score) in enumerate(_retrieve_jobs(num_seqs_to_translate)):
        # TODO: hide trimming of the 'None' at the beginning of each translation within search or decoder
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

    return output_file_name


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parallel_iterator(source_file, target_file):
    with codecs.open(source_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trg:
            for src_l, trg_l in itertools.izip(src, trg):
                yield (src_l, trg_l)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",
                        help="the source text for the test corpus")
    parser.add_argument("-t", "--target",
                        help="the target text for the test corpus")
    parser.add_argument("-c", "--config", help="the yaml config file for the IMT system")
    parser.add_argument("-o", "--outputdir",
                        help="the directory where we should write the output files and the experiment report")
    parser.add_argument("-p", "--parallel", type=int,
                        help="the number of models to use in parallel")
    parser.add_argument("-nc", "--numcycles", type=int,
                        help="the number of PRIMT cycles to perform")
    args = parser.parse_args()
    arg_dict = vars(args)

    num_procs = 10


    score_file = os.path.join(args.outputdir, 'iteration_scores.BLEU')

    # overwrite old version if it exists
    if os.path.exists(score_file):
        logger.warn('{} already exists, moving to: {}'.format(score_file, score_file + '.old'))
        shutil.copyfile(score_file, score_file + '.old')
        open(score_file, 'w')

    # start with nothing
    all_cycle_constraints = [[] for _ in codecs.open(args.source).read().strip().split('\n')]

    for cycle_idx in range(args.numcycles):
        logger.info('Running PRIMT cycle: {}'.format(cycle_idx))
        primt_output_file = run_primt_cycle(args.source, args.target, args.config, args.outputdir, cycle_idx,
                                            all_cycle_constraints, num_parallel=args.parallel)

        cycle_constraints = create_constraints(primt_output_file, args.target)
        # add these constraints for the next cycle
        for cons_i, cons in enumerate(cycle_constraints):
            # Note if we add empty constraints decoding will break
            if len(cons) > 0:
                all_cycle_constraints[cons_i].append(cons)

        cycle_bleu = compute_bleu_score(primt_output_file, args.target)
        logger.info("CYCLE: {} BLEU: {}".format(cycle_idx, cycle_bleu))
        with codecs.open(score_file, 'a', encoding='utf8') as scores:
            scores.write("CYCLE: {} BLEU: {}".format(cycle_idx, cycle_bleu))

        with codecs.open(os.path.join(args.outputdir, 'cycle_constraints.{}.json'.format(cycle_idx)),
                         'w', encoding='utf8') as cons_out:
            cons_out.write(json.dumps(cycle_constraints, indent=2))

    import ipdb;ipdb.set_trace()

    # TODO: fix constraints to be real ones
    # fake_constraints = [[] for _ in range(len(open(source_file).read().strip().split('\n')))]

    # TODO: add a sanity check that hyps and refs have the same number of lines, and no refs or hyps are empty
    # get gold refs


    # compute BLEU and write to experiment log
    # with codecs.open(os.path.join(score_file, 'a')) as out:

    # WORKING: get output, extract constraints, then go again
    logger.info('Finished translating {0} with constraints'.format(args.source))
    # logger.info('Files in {}: {}'.format(args.outputdir, os.listdir(args.outputdir)))


    # logger.info("Validation Took: {} minutes".format(
    #     float(time.time() - val_start_time) / 60.))
