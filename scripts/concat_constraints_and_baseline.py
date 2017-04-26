"""
Script for sanity checking constraint placement: prepend constraints at the beginning of the baseline translation

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
from subprocess import Popen, PIPE


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BLEU_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/multi-bleu.perl')

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


def concat_constraints(translation_file, output_dir, constraints):
    mkdir_p(output_dir)
    output_file_name = os.path.join(output_dir, 'translations.out')

    with codecs.open(translation_file, encoding='utf8') as inp:
        with codecs.open(output_file_name, 'a', encoding='utf8') as out:
            for line_idx, (trans, cons) in enumerate(itertools.izip(inp, constraints)):
                trans = trans.strip()
                cons_str = u' '.join([t for c in cons for t in c])
                trans = cons_str + u' ' + trans
                out.write(trans + u'\n')

    return output_file_name


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
    parser.add_argument("-t", "--translations",
                        help="the baseline translation output")
    parser.add_argument("--constraints", help="A json file containing a list of constraints for each segment")
    parser.add_argument("-o", "--outputdir",
                        help="the directory where we should write the output files and the experiment report")

    args = parser.parse_args()
    arg_dict = vars(args)

    score_file = os.path.join(args.outputdir, 'experiment_scores.BLEU')

    # overwrite old version if it exists
    if os.path.exists(score_file):
        logger.warn('{} already exists, moving to: {}'.format(score_file, score_file + '.old'))
        shutil.copyfile(score_file, score_file + '.old')
        open(score_file, 'w')

    constraints = json.loads(codecs.open(args.constraints, encoding='utf8').read())

    logger.info('Appending constraints in {} to {}'.format(args.translations, args.constraints))
    output_file_name = concat_constraints(args.translations, args.outputdir, constraints)

    output_bleu = compute_bleu_score(output_file_name, args.target)
    logger.info("BLEU: {}".format(output_bleu))
    with codecs.open(score_file, 'a', encoding='utf8') as scores:
        scores.write("BLEU: {}".format(output_bleu))

    logger.info('Finished'.format(args.source))


