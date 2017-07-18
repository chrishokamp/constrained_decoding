"""
Translate an input file (optionally with constraints), using one or more Nematus models,
optionally specifying a weight for each model.
"""

import argparse
import logging
import sys
import json
import codecs
import itertools
import os

from constrained_decoding import create_constrained_decoder
from constrained_decoding.translation_model.nematus_tm import NematusTranslationModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_config(filename):
    # defaults -- params which are inputs to `nematus/translate.py`, but aren't part of the persisted *.json config
    translate_config = {
        "return_alignment": False
    }
    config = json.loads(codecs.open(filename, encoding='utf8').read())
    return dict(translate_config, **config)


def decode(decoder, translation_model, inputs, n_best, max_hyp_len, beam_size=5,
           constraints=None,
           mert_nbest=False,
           return_alignments=False,
           length_norm=True):

    mapped_inputs = translation_model.map_inputs(inputs)

    input_constraints = []
    if constraints is not None:
        input_constraints = translation_model.map_constraints(constraints)

    start_hyp = translation_model.start_hypothesis(mapped_inputs, input_constraints)

    # Note: the length_factor is used with the length of the first model input of the ensemble
    search_grid = decoder.search(start_hyp=start_hyp,
                                 constraints=input_constraints,
                                 max_hyp_len=max_hyp_len,
                                 beam_size=beam_size)

    best_output, best_alignments = decoder.best_n(search_grid, translation_model.eos_token,
                                                  n_best=n_best,
                                                  return_model_scores=mert_nbest,
                                                  return_alignments=return_alignments,
                                                  length_normalization=length_norm)

    if return_alignments:
        return best_output, best_alignments
    else:
        return best_output


def run(input_files, constraints_file, output, models, configs, weights,
        n_best=1, length_factor=1.3, beam_size=5, mert_nbest=False, write_alignments=None, length_norm=True):

    if configs is not None:
        assert len(models) == len(configs), 'Number of models differs from numer of config files'

    if weights is not None:
        assert len(models) == len(weights), 'If you specify weights, there must be one for each model'

    return_alignments = False
    if write_alignments is not None:
        return_alignments = True
        try:
            os.remove(write_alignments)
        except OSError:
            pass

    # remember Nematus needs _encoded_ utf8
    if configs is not None:
        configs = [load_config(f) for f in configs]

    # build ensembled TM
    nematus_tm = NematusTranslationModel(models, configs, model_weights=weights)

    # Build GBS search
    decoder = create_constrained_decoder(nematus_tm)

    constraints = None
    if constraints_file is not None:
        constraints = json.loads(codecs.open(constraints_file, encoding='utf8').read())

    if output.name != '<stdout>':
        output = codecs.open(output.name, 'w', encoding='utf8')

    input_iters = []
    for input_file in input_files:
        input_iters.append(codecs.open(input_file, encoding='utf8'))

    for idx, inputs in enumerate(itertools.izip(*input_iters)):
        input_constraints = []
        if constraints is not None:
            input_constraints = constraints

        # Note: the length_factor is used with the length of the first model input of the ensemble
        # in case the users constraints will go beyond the max length according to length_factor
        max_hyp_len = int(round(len(inputs[0].split()) * length_factor))
        if len(input_constraints) > 0:
            num_constraint_tokens = sum(1 for c in input_constraints for _ in c)
            if num_constraint_tokens >= max_hyp_len:
                logger.warn('The number of tokens in the constraints are greater than max_len*length_factor, ' + \
                            'autoscaling the maximum hypothesis length...')
                max_hyp_len = num_constraint_tokens + int(round(len(max_hyp_len) / 2))

        best_output = decode(decoder, nematus_tm, inputs, n_best,
                             max_hyp_len=max_hyp_len,
                             beam_size=beam_size,
                             constraints=input_constraints,
                             return_alignments=return_alignments,
                             length_norm=length_norm)

        if return_alignments:
            # decoding returned a tuple with 2 items
            best_output, best_alignments = best_output
        
        if n_best > 1:
            if mert_nbest:
                # format each n-best entry in the mert format
                translations, scores, model_scores = zip(*best_output)
                # start from idx 1 to cut off `None` at the beginning of the sequence
                translations = [u' '.join(s[1:]) for s in translations]
                # create dummy feature names
                model_names = [u'M{}'.format(m_i) for m_i in range(len(model_scores[0]))]
                #Note: we make model scores and logprob negative for MERT optimization to work
                model_score_strings = [u' '.join([u'{}= {}'.format(model_name, -s_i)
                                                  for model_name, s_i in zip(model_names, m_scores)])
                                       for m_scores in model_scores]
                nbest_output_strings = [u'{} ||| {} ||| {} ||| {}'.format(idx, translation, feature_scores, -logprob)
                                        for translation, feature_scores, logprob
                                        in zip(translations, model_score_strings, scores)]
                decoder_output = u'\n'.join(nbest_output_strings) + u'\n'
            else:
                # start from idx 1 to cut off `None` at the beginning of the sequence
                # separate each n-best list with newline
                decoder_output = u'\n'.join([u' '.join(s[0][1:]) for s in best_output]) + u'\n\n'

            if output.name == '<stdout>':
                output.write(decoder_output.encode('utf8'))
            else:
                output.write(decoder_output)
        else:
            # start from idx 1 to cut off `None` at the beginning of the sequence
            decoder_output = u' '.join(best_output[0][1:])
            if output.name == '<stdout>':
                output.write((decoder_output + u'\n').encode('utf8'))
            else:
                output.write(decoder_output + u'\n')

        # Note alignments are always an n-best list (may be n=1)
        if write_alignments is not None:
            with codecs.open(write_alignments, 'a+', encoding='utf8') as align_out:
                align_out.write(json.dumps([a.tolist() for a in best_alignments]) + u'\n')

        if (idx+1) % 10 == 0:
            logger.info('Wrote {} translations to {}'.format(idx+1, output.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: support yaml configs
    parser.add_argument('-m', '--models', nargs='+', help='the paths to one or more Nematus models', required=True)
    parser.add_argument('-c', '--configs', nargs='+', help='paths to one or more config.json files for Nematus models',
                        default=None, required=False)
    parser.add_argument('--weights', nargs='+', type=float, default=None, required=False,
                        help='(Optional) one weight per model, will be applied to `log(p_model)` at each timestep')
    parser.add_argument('--constraints', type=str, default=None, required=False,
                        help='(Optional) json file containing one (possibly empty) list of constraints per input line')
    parser.add_argument('--beam_size', type=int, default=5, required=False,
                        help='Decoder beam size (default=5)')
    parser.add_argument('--nbest', type=int, default=1, required=False,
                        help='N-best list size, must be 1 <= nbest <= beam_size')
    parser.add_argument('--mert_nbest', dest='mert_nbest', action='store_true',
                        help='If you use this argument, n-best list will be printed in format used for Moses MERT')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='(Optional) a file name in MERT *.dense format which specifies the weights for each model')
    parser.add_argument('--length_factor', type=float, default=1.3,
                        help='(Optional) the factor to multiply the first input by to get the maximum output length for decoding')
    parser.add_argument('--alignments_output', default=None,
                        help='(Optional) if a string is provided, alignment weights will be written to this file')
    parser.add_argument('--no_length_norm', dest='length_norm', action='store_false',
                        help='(Optional) if --no_length_norm is included, scores will not be normalized by hyp length')
    parser.set_defaults(mert_nbest=False)
    parser.set_defaults(length_norm=True)
    parser.add_argument('-i', '--inputs', nargs='+', help="one or more input text files, corresponding to each model")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout,
                        help="Where to write the translated output")
    args = parser.parse_args()

    assert 1 <= args.nbest <= args.beam_size, 'N-best size must be 1 <= nbest <= args.beam_size'

    # if user specified a weights file, assert that they didn't also specify weights on the command line
    if args.weights is not None and args.load_weights is not None:
        raise AssertionError('only one of {weights, load_weights} should be specified')

    if args.load_weights is not None:
        with codecs.open(args.load_weights, encoding='utf8') as weights_file:
            args.weights = [float(l.strip().split()[-1]) for l in weights_file]

    run(args.inputs, args.constraints, args.output, args.models, args.configs, args.weights,
        n_best=args.nbest, beam_size=args.beam_size, mert_nbest=args.mert_nbest, length_factor=args.length_factor,
        write_alignments=args.alignments_output, length_norm=args.length_norm)
