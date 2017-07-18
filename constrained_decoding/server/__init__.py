import logging
import os
import re
import json
import os
import codecs
from subprocess import Popen, PIPE

from flask import Flask, request, render_template, jsonify, abort

from constrained_decoding import create_constrained_decoder
# from wtforms import Form, TextAreaField, validators

logger = logging.getLogger(__name__)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.models = None

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(path_to_this_dir, 'templates')


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
    This class encapusulates pre- and post-processing functionality for TermNMT workflows
    
    """

    def __init__(self, source_lang, use_subword=False, subword_codes=None):
        self.use_subword = use_subword
        if self.use_subword:
            subword_codes_iter = codecs.open(subword_codes, encoding='utf-8')
            self.bpe = BPE(subword_codes_iter)

        self.source_lang = source_lang

        # Note hardcoding of script location within repo
        tokenize_script = os.path.join(os.path.dirname(__file__), 'resources/tokenizer/tokenizer.perl')
        self.tokenizer_cmd = [tokenize_script, '-l', self.source_lang, '-no-escape', '1', '-q', '-', '-b']
        self.tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE, bufsize=1)

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
        pass

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


    def map_terms(self, tokens):
        """
        Map tokenized string through terminology
        
        Args:
          tokens: 
          
        Returns:
          
        """
        pass


# class NeuralMTDemoForm(Form):
#     sentence = TextAreaField('Write the sentence here:',
#                              [validators.Length(min=1, max=100000)])
#     prefix = TextAreaField('Write the target prefix here:',
#                              [validators.Length(min=1, max=100)])
#     target_text = ''


# TODO: multiple instances of the same model, delegate via thread queue?
# TODO: supplementary info via endpoint -- softmax confidence, cumulative confidence, etc...
# TODO: online updating via cache
# TODO: require source and target language specification
# @app.route('/neural_MT_demo', methods=['GET', 'POST'])
# def neural_mt_demo():
#     form = NeuralMTDemoForm(request.form)
#     if request.method == 'POST' and form.validate():
#
#         # TODO: delegate to prefix decoder function here
#         source_text = form.sentence.data # Problems in Spanish with 'A~nos. E'
#         target_text = form.prefix.data
#         logger.info('Acquired lock')
#         lock.acquire()
#
#         source_sentence = source_text.encode('utf-8')
#         target_prefix = target_text.encode('utf-8')
#
#         # translations, costs = app.predictor.predict_segment(source_sentence, tokenize=True, detokenize=True)
#         translations, costs = app.predictor.predict_segment(source_sentence, target_prefix=target_prefix,
#                                                             tokenize=True, detokenize=True, n_best=10)
#
#         output_text = ''
#         for hyp in translations:
#             output_text += hyp + '\n'
#
#         form.target_text = output_text.decode('utf-8')
#         logger.info('detokenized translations:\n {}'.format(output_text))
#
#         print "Lock release"
#         lock.release()
#
#     return render_template('neural_MT_demo.html', form=form)


@app.route('/translate', methods=['GET', 'POST'])
def neural_mt_endpoint():
    # TODO: parse request object, remove form
    if request.method == 'POST':
        request_data = request.get_json()
        print(request_data)
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

        translations = decode(source_lang, target_lang, source_sentence,
                              constraints=target_constraints, n_best=n_best)

        #print "Lock release"
        #lock.release()

    return jsonify({'ranked_translations': translations})


# TODO: add DataProcessor initialized with all the assets we need for pre-/post- processing
# TODO: add terminology mapping hooks into Terminology NMT DataProcessor
# TODO: Map and unmap hooks
# TODO: remember that placeholder term mapping needs to pass the restore map through to postprocessing
# TODO: restoring @num@ placeholders with word alignments? -- leave this for last


def decode(source_lang, target_lang, source_sentence, constraints=None, n_best=1, length_factor=1.5, beam_size=5):
    """
    Decode an input sentence
    
    Args:
      source_lang: two char src lang abbreviation
      target_lang: two char src lang abbreviation
      source_sentence: the source sentence to translate (we assume already preprocessed)
      n_best: the length of the n-best list to return (default=1)
      
    Returns:
        
    """

    model = app.models[(source_lang, target_lang)]
    decoder = app.decoders[(source_lang, target_lang)]
    # Note: remember we support multiple inputs for each model (i.e. each model may be an ensemble where sub-models
    # Note: accept different inputs)

    data_processor = app.processors.get((source_lang, target_lang), None)
    if data_processor is not None:
        source_sentence = u' '.join(data_processor.tokenize(source_sentence))

    inputs = [source_sentence]

    mapped_inputs = model.map_inputs(inputs)

    input_constraints = []
    if constraints is not None:
        input_constraints = [data_processor.tokenize(c) for c in constraints]
        input_constraints = model.map_constraints(input_constraints)

    start_hyp = model.start_hypothesis(mapped_inputs, input_constraints)

    beam_size = max(n_best, beam_size)
    search_grid = decoder.search(start_hyp=start_hyp, constraints=input_constraints,
                                 max_hyp_len=int(round(len(mapped_inputs[0][0]) * length_factor)),
                                 beam_size=beam_size)

    best_output, best_alignments = decoder.best_n(search_grid, model.eos_token, n_best=n_best,
                                                  return_model_scores=False, return_alignments=True,
                                                  length_normalization=True)

    # WORKING: get the constraint indices from the hypothesis and return these as well
    # TODO: logic for k-best vs 1-best
    best_hyp = best_output[-1]
    # import ipdb; ipdb.set_trace()

    if n_best > 1:
        # start from idx 1 to cut off `None` at the beginning of the sequence
        # separate each n-best list with newline
        decoder_output = [u' '.join(s[0][1:]) for s in best_output]
    else:
        # start from idx 1 to cut off `None` at the beginning of the sequence
        decoder_output = [u' '.join(best_output[0][1:])]

    # Note alignments are always an n-best list (may be n=1)
    # if write_alignments is not None:
    #     with codecs.open(write_alignments, 'a+', encoding='utf8') as align_out:
    #         align_out.write(json.dumps([a.tolist() for a in best_alignments]) + u'\n')
    return decoder_output

    # best_n_hyps, best_n_costs, best_n_glimpses, best_n_word_level_costs, best_n_confidences, src_in = predictor.predict_segment(source_sentence, target_prefix=target_prefix,
    #                                                     tokenize=True, detokenize=True, n_best=n_best, max_length=predictor.max_length)

    # TODO: we _must_ add subword configuration as well -- we need to apply subword, then re-concat afterwards
    # remove EOS and normalize subword
    # def _postprocess(hyp):
    #     hyp = re.sub("</S>$", "", hyp)
    #     # Note the order of the next two lines is important
    #     hyp = re.sub("\@\@ ", "", hyp)
    #     hyp = re.sub("\@\@", "", hyp)
    #     return hyp
    #
    # postprocessed_hyps = [_postprocess(h) for h in best_n_hyps]

    # return postprocessed_hyps


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

