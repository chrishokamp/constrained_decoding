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

import logging
import argparse
import codecs
import itertools
import errno
import cPickle
import os
from collections import Counter
from multiprocessing import Process, Queue

import numpy as np

from constrained_decoding.translation_model.nmt import NeuralTranslationModel
from constrained_decoding import ConstrainedDecoder, Beam

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



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
def main(source_file, target_file, config_file, output_dir, constraints=None, num_parallel=1):

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

    # TODO: fix constraints to be real ones
    fake_constraints = [[] for _ in range(len(open(source_file).read().strip().split('\n')))]
    num_seqs_to_translate, source_sens, target_sens = _send_jobs(source_file, target_file, fake_constraints)
    _finish_processes()

    for i, trans in enumerate(_retrieve_jobs(num_seqs_to_translate)):
        import ipdb;ipdb.set_trace()


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



# create models
# for model in models
# start model, pass as arg to the function which is the target of the Process


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

    main(args.source, args.target, args.config, args.outputdir, num_parallel=args.parallel)

    logger.info('Finished translating {0} with constraints'.format(args.source))
    # logger.info('Files in {}: {}'.format(args.outputdir, os.listdir(args.outputdir)))









