import logging
import os
import codecs
import re
from subprocess import Popen, PIPE

from flask import Flask, request, render_template, jsonify, abort

from constrained_decoding import create_constrained_decoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.models = None

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(path_to_this_dir, 'templates')


def remap_constraint_indices(tokenized_sequence, detokenized_sequence, constraint_indices):
    """
    Map the constraint indices of a tokenized sequence to the indices of a detokenized sequence

    Any time there was '@@ ' in the tokenized sequence, we removed it
      - the detokenized sequence has fewer spaces than the tokenized sequence
    """
    constraint_idx_starts = {start: end for start, end in constraint_indices}
    constraint_idx_ends = {end: start for start, end in constraint_indices}

    remapped_indices = []
    tokenized_idx = 0
    current_offset = 0
    true_start = None
    for true_idx, output_char in enumerate(detokenized_sequence):
        if tokenized_idx in constraint_idx_starts:
            true_start = tokenized_idx - current_offset
        elif tokenized_idx in constraint_idx_ends:
            assert true_start is not None, 'if we found an end, we also need a start'
            true_end = tokenized_idx - current_offset
            remapped_indices.append([true_start, true_end])
            true_start = None
        # this logic assumes that post-processing did not _change_ any characters
        # I.e. no characters were substituted for other characters
        while output_char != tokenized_sequence[tokenized_idx]:
            tokenized_idx += 1
            current_offset += 1
            if tokenized_idx > len(tokenized_sequence):
                raise IndexError('We went beyond the end of the longer sequence: {}, when comparing with: {}'.format(
                    tokenized_sequence,
                    detokenized_sequence
                ))

            if tokenized_idx in constraint_idx_starts:
                true_start = tokenized_idx - current_offset
            elif tokenized_idx in constraint_idx_ends:
                assert true_start is not None, 'if we found an end, we also need a start'
                true_end = tokenized_idx - current_offset
                remapped_indices.append([true_start, true_end])
                true_start = None

        tokenized_idx += 1

    if true_start is not None:
        true_end = tokenized_idx - current_offset
        remapped_indices.append([true_start, true_end])

    return remapped_indices


def convert_token_annotations_to_spans(token_sequence, constraint_annotations):
    print('len tokens: {}'.format(len(token_sequence)))
    print('len annotations: {}'.format(len(constraint_annotations)))
    import ipdb;ipdb.set_trace()
    assert len(token_sequence) == len(constraint_annotations), 'we need one annotation per token for this to make sense'
    # here we are just annotating which spans are constraints, we discard the constraint alignment information

    span_annotations = []
    output_sequence = u''
    constraint_id = None
    constraint_start_idx = None
    for token, annotation in zip(token_sequence, constraint_annotations):

        if annotation is not None:
            if annotation[0] != constraint_id:
                # we're starting a new constraint
                constraint_start_idx = len(output_sequence)
                if len(output_sequence) > 0:
                    # we'll add a whitespace before the constraint starts below
                    constraint_start_idx += 1

                constraint_id = annotation[0]
        else:
            # a constraint just finished
            if constraint_id is not None:
                span_annotations.append([constraint_start_idx, len(output_sequence)])
                constraint_id = None
                constraint_start_idx = None

        if len(output_sequence) == 0:
            output_sequence = token
        else:
            output_sequence = u'{} {}'.format(output_sequence, token)


    return span_annotations, output_sequence


def get_pairs(word):
    """ (Subword Encoding) Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, cache=None):
    """
    (Subword Encoding) Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if cache is None:
        cache = {}

    if orig in cache:
        return cache[orig]

    word = tuple(orig) + ('</w>',)
    pairs = get_pairs(word)

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    cache[orig] = word
    return word


class BPE(object):

    def __init__(self, codes, separator='@@', ignore=None):
        self.bpe_codes = [tuple(item.split()) for item in codes]
        self.ignore = ignore

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])
        self.separator = separator

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = []
        for word in sentence.split():
            if self.ignore is not None and word in self.ignore:
                output.append(word)
            else:
                new_word = encode(word, self.bpe_codes)

                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])

        return u' '.join(output)


class DataProcessor(object):
    """
    This class encapusulates pre- and post-processing functionality

    """

    def __init__(self, lang, use_subword=False, subword_codes=None):
        self.use_subword = use_subword
        if self.use_subword:
            subword_codes_iter = codecs.open(subword_codes, encoding='utf-8')
            self.bpe = BPE(subword_codes_iter)

        self.lang = lang

        # Note hardcoding of script location within repo
        tokenize_script = os.path.join(os.path.dirname(__file__), 'resources/tokenizer/tokenizer.perl')
        self.tokenizer_cmd = [tokenize_script, '-l', self.lang, '-no-escape', '1', '-q', '-', '-b']
        self.tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE, bufsize=1)

        detokenize_script = os.path.join(os.path.dirname(__file__), 'resources/tokenizer/detokenizer.perl')
        self.detokenizer_cmd = [detokenize_script, '-l', self.lang, '-q', '-']

    def tokenize(self, text):
        if len(text.strip()) == 0:
            return []

        if type(text) is unicode:
            text = text.encode('utf8')
        self.tokenizer.stdin.write(text + '\n\n')
        self.tokenizer.stdin.flush()
        self.tokenizer.stdout.flush()

        # this logic is due to issues with calling out to the moses tokenizer
        segment = '\n'
        while segment == '\n':
            segment = self.tokenizer.stdout.readline()
        # read one more line
        _ = self.tokenizer.stdout.readline()

        utf_line = segment.rstrip().decode('utf8')

        if self.use_subword:
            tokens = self.bpe.segment(utf_line).split()
        else:
            tokens = utf_line.split()
        return tokens

    def detokenize(self, text):
        """
        Detokenize a string using the moses detokenizer

        Args:

        Returns:

        """
        if self.use_subword:
            text = re.sub("\@\@ ", "", text)
            text = re.sub("\@\@", "", text)

        if type(text) is unicode:
            text = text.encode('utf8')

        detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
        text, _ = detokenizer.communicate(text)

        utf_line = text.rstrip().decode('utf8')
        return utf_line

    def truecase(self, text):
        """
        Truecase a string with this DataProcessor's truecasing model

        Args:

        Returns:

        """
        pass

    def detruecase(self, text):
        """
        Deruecase a string using moses detruecaser

        Args:

        Returns:

        """
        pass


# TODO: multiple instances of the same model, delegate via thread queue? -- with Flask this is way too buggy
# TODO: online updating via cache
# TODO: require source and target language specification
@app.route('/translate', methods=['POST'])
def constrained_decoding_endpoint():
    request_data = request.get_json()
    logger.info('Request data: {}'.format(request_data))
    source_lang = request_data['source_lang']
    target_lang = request_data['target_lang']
    n_best = request_data.get('n_best', 1)

    if (source_lang, target_lang) not in app.models:
        logger.error('MT Server does not have a model for: {}'.format((source_lang, target_lang)))
        abort(404)

    source_sentence = request_data['source_sentence']
    target_constraints = request_data.get('target_constraints', None)

    #logger.info('Acquired lock')
    #lock.acquire()
    #print "Lock release"
    #lock.release()

    # Note best_hyps is always a list
    best_outputs = decode(source_lang, target_lang, source_sentence,
                       constraints=target_constraints, n_best=n_best)

    target_data_processor = app.processors.get(target_lang, None)

    # each output is (seq, score, true_len, hypothesis)
    output_objects = []
    for seq, score, hyp in best_outputs:
        # start from 1 to cut off the start symbol (None)
        true_len = int(hyp.true_len)
        span_annotations, raw_hyp = convert_token_annotations_to_spans(seq[1:true_len],
                                                                       hyp.constraint_indices[1:true_len])

        detokenized_hyp = target_data_processor.detokenize(raw_hyp)

        # map tokenized constraint indices to post-processed sequence indices
        detokenized_span_indices = remap_constraint_indices(tokenized_sequence=raw_hyp,
                                                            detokenized_sequence=detokenized_hyp,
                                                            constraint_indices=span_annotations)
        # print(u"raw hyp: {}".format(raw_hyp))
        # print(u"detokenized hyp: {}".format(detokenized_hyp))
        # print(u"tokenized indices: {}".format(span_annotations))
        # print(u"detokenized indices: {}".format(detokenized_span_indices))

        output_objects.append({'translation': detokenized_hyp,
                               'constraint_annotations': detokenized_span_indices,
                               'score': score})


    return jsonify({'outputs': output_objects})


def decode(source_lang, target_lang, source_sentence, constraints=None, n_best=1, length_factor=1.5, beam_size=5):
    """
    Decode an input sentence

    Args:
      source_lang: two-char src lang abbreviation
      target_lang: two-char target lang abbreviation
      source_sentence: the source sentence to translate (we assume already preprocessed)
      n_best: the length of the n-best list to return (default=1)

    Returns:

    """

    model = app.models[(source_lang, target_lang)]
    decoder = app.decoders[(source_lang, target_lang)]
    # Note: remember we support multiple inputs for each model (i.e. each model may be an ensemble where sub-models
    # Note: accept different inputs)

    source_data_processor = app.processors.get(source_lang, None)
    target_data_processor = app.processors.get(target_lang, None)

    if source_data_processor is not None:
        source_sentence = u' '.join(source_data_processor.tokenize(source_sentence))

    inputs = [source_sentence]

    mapped_inputs = model.map_inputs(inputs)

    input_constraints = []
    if constraints is not None:
        if target_data_processor is not None:
            input_constraints = [target_data_processor.tokenize(c) for c in constraints]

        input_constraints = model.map_constraints(input_constraints)

    start_hyp = model.start_hypothesis(mapped_inputs, input_constraints)

    beam_size = max(n_best, beam_size)
    search_grid = decoder.search(start_hyp=start_hyp, constraints=input_constraints,
                                 max_hyp_len=int(round(len(mapped_inputs[0][0]) * length_factor)),
                                 beam_size=beam_size)

    best_output, best_alignments = decoder.best_n(search_grid, model.eos_token, n_best=n_best,
                                                  return_model_scores=False, return_alignments=True,
                                                  length_normalization=True)

    if n_best == 1:
        best_output = [best_output]

    return best_output


# Note: this function will break libgpuarray if theano is using the GPU
def run_imt_server(models, processors=None, port=5007):
    # Note: servers use a special .yaml config format-- maps language pairs to NMT configuration files
    # the server instantiates a predictor for each config, and hashes them by language pair tuples -- i.e. (en,fr)
    # Caller passes in a dict of predictors, keys are tuples (source_lang, target_lang)
    if processors is None:
        app.processors = {k: None for k in models.keys()}
    else:
        app.processors = processors

    app.models = models
    app.decoders = {k: create_constrained_decoder(v) for k, v in models.items()}


    logger.info('Server starting on port: {}'.format(port))
    # logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
    # app.run(debug=True, port=port, host='127.0.0.1', threaded=True)
    app.run(debug=True, port=port, host='127.0.0.1', threaded=False)

