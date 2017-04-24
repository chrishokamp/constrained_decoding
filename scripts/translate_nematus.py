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


def run(input_files, constraints_file, output, models, configs, weights, n_best=1, length_factor=1.3, beam_size=5):

    assert len(models) == len(configs), 'We need one config file for every model'
    if weights is not None:
        assert len(models) == len(weights), 'If you specify weights, there must be one for each model'


    # TODO: remember Nematus needs _encoded_ utf8
    # TODO: remember paths to vocab dictionaries may not be fully specified
    # TODO: remember multiple input files

    configs = [load_config(f) for f in configs]

    # build ensembled TM
    nematus_tm = NematusTranslationModel(models, configs, model_weights=weights)

    # Build GBS search
    decoder = create_constrained_decoder(nematus_tm)

    constraints = None
    if constraints_file is not None:
        constraints = json.loads(codecs.open(constraints_file, encoding='utf8').read())

    if output.name != '<stdout>':
        output = codecs.open(output.name, 'w', encoding='utf-8')

    input_iters = []
    for input_file in input_files:
        input_iters.append(codecs.open(input_file, encoding='utf8'))

    for idx, inputs in enumerate(itertools.izip(*input_iters)):
        mapped_inputs = nematus_tm.map_inputs(inputs)

        input_constraints = []
        if constraints is not None:
            input_constraints = nematus_tm.map_constraints(constraints[idx])

        start_hyp = nematus_tm.start_hypothesis(mapped_inputs, input_constraints)

        search_grid = decoder.search(start_hyp=start_hyp, constraints=input_constraints,
                                     max_hyp_len=int(round(len(mapped_inputs[0][0]) * length_factor)),
                                     beam_size=beam_size)
        best_output = decoder.best_n(search_grid, nematus_tm.eos_token, n_best=n_best)

        if n_best > 1:
            # start from idx 1 to cut off `None` at the beginning of the sequence
            decoder_outputs = [u' '.join(s[0][1:]) for s in best_output]
            # separate each n-best list with newline
            output.write(u'\n'.join(decoder_outputs) + u'\n\n')
        else:
            decoder_output = u' '.join(best_output[0][1:])
            output.write(decoder_output + u'\n')

        if idx+1 % 10 == 0:
            logger.info('Wrote {} translations to {}'.format(idx+1, output.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: support yaml configs
    parser.add_argument('-m', '--models', nargs='+', help='the paths to one or more Nematus models', required=True)
    parser.add_argument('-c', '--configs', nargs='+', help='paths to one or more config.json files for Nematus models',
                        required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=None, required=False,
                        help='(Optional) one weight per model, will be applied to `log(p_model)` at each timestep')
    parser.add_argument('--constraints', type=str, default=None, required=False,
                        help='(Optional) json file containing one (possibly empty) list of constraints per input line')
    parser.add_argument('-i', '--inputs', nargs='+', help="one or more input text files, corresponding to each model")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout,
                        help="Where to write the translated output")
    args = parser.parse_args()

    run(args.inputs, args.constraints, args.output, args.models, args.configs, args.weights)




# TODO: average model scores, or weighted sum
# TODO: assert that dictionaries were found -- i.e. full paths to dictionaries are correct

