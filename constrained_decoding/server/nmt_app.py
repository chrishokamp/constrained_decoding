import logging
import copy
import json
from flask import Flask, request, jsonify, abort

import pylru

from constrained_decoding import create_constrained_decoder
from constrained_decoding.server import convert_token_annotations_to_spans, remap_constraint_indices

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.models = None

cache_size = 1000
app.local_cache = pylru.lrucache(cache_size)


# TODO: multiple instances of the same model, delegate via thread queue? -- with Flask this is way too buggy
# TODO: online updating via cache
@app.route('/translate', methods=['POST'])
def constrained_decoding_endpoint():
    request_data = request.get_json()
    logger.info('Request data: {}'.format(request_data))
    source_lang = request_data['source_lang']
    target_lang = request_data['target_lang']
    n_best = request_data.get('n_best', 1)
    
    beam_size = 1

    if (source_lang, target_lang) not in app.models:
        logger.error('MT Server does not have a model for: {}'.format((source_lang, target_lang)))
        abort(404)

    source_sentence = request_data['source_sentence']
    target_constraints = request_data.get('target_constraints', None)

    model = app.models[(source_lang, target_lang)]
    decoder = app.decoders[(source_lang, target_lang)]

    cache_obj = copy.deepcopy(request_data)
    del cache_obj['request_time']
    cache_str = json.dumps(cache_obj)
    if cache_str in app.local_cache:
        logger.info('Cache hit: {}'.format(cache_str))
        return app.local_cache[cache_str]

    # Note: remember we support multiple inputs for each model (i.e. each model may be an ensemble where sub-models
    # Note: accept different inputs)

    source_data_processor = app.processors.get(source_lang, None)
    target_data_processor = app.processors.get(target_lang, None)

    logger.info('map source')
    if source_data_processor is not None:
        source_sentence = u' '.join(source_data_processor.tokenize(source_sentence))

    logger.info('map constraints')
    if target_constraints is not None:
        if target_data_processor is not None:
            # hack to avoid truecasing constraints
            temp_truecase = target_data_processor.truecase
            target_data_processor.truecase = False
            target_constraints = [target_data_processor.tokenize(c) for c in target_constraints]
            target_data_processor.truecase = temp_truecase

    logger.info('decode')
    # Note best_hyps is always a list
    best_outputs = decode(source_sentence,  model, decoder,
                          constraints=target_constraints, n_best=n_best, beam_size=beam_size)

    target_data_processor = app.processors.get(target_lang, None)

    # each output is (seq, score, true_len, hypothesis)
    output_objects = []
    for seq, score, hyp in best_outputs:
        # start from 1 to cut off the start symbol (None)
        true_len = int(hyp.true_len)

        #logger.info('BEST HYP BEFORE TRUNCATION: {}'.format(hyp.sequence))
        #logger.info('True len: {}'.format(true_len))
        #logger.info('Seq: {}'.format(seq))

        # this is a hack to make sure escaped punctuation gets matched correctly
        # Note this is way too slow, we need another solution
        if target_data_processor.escape_special_chars:
            # detokenized_hyp = target_data_processor.deescape_special_chars(detokenized_hyp)
            seq = [target_data_processor.deescape_special_chars(tok) if tok is not None else tok
                   for tok in seq]

        span_annotations, raw_hyp = convert_token_annotations_to_spans(seq[1:true_len],
                                                                       hyp.constraint_indices[1:true_len])

        # detokenization also de-escapes
        detokenized_hyp = target_data_processor.detokenize(raw_hyp)

        # here we do the punctuation denormalization
        # map tokenized constraint indices to post-processed sequence indices
        detokenized_span_indices = remap_constraint_indices(tokenized_sequence=raw_hyp,
                                                            detokenized_sequence=detokenized_hyp,
                                                            constraint_indices=span_annotations)

        # finally detruecase
        if target_data_processor.truecase:
        #     detokenized_hyp = target_data_processor.detruecase(detokenized_hyp)
            # just a hack to make sure the capitalization of the mapped target and the original target matches
            detokenized_hyp = detokenized_hyp[0].upper() + detokenized_hyp[1:]

        output_objects.append({'translation': detokenized_hyp,
                               'constraint_annotations': detokenized_span_indices,
                               'score': score})

    output_json = jsonify({'outputs': output_objects})
    app.local_cache[cache_str] = output_json

    return output_json


def decode(source_sentence, model, decoder,
           constraints=None, n_best=1, length_factor=1.3, beam_size=5):
    """
    Decode an input sentence

    Args:
      source_lang: two-char src lang abbreviation
      target_lang: two-char target lang abbreviation
      source_sentence: the source sentence to translate (we assume already preprocessed)
      n_best: the length of the n-best list to return (default=1)

    Returns:

    """

    # we wrap in a list because this server doesn't support multiple inputs, but the decoder does
    inputs = [source_sentence]

    mapped_inputs = model.map_inputs(inputs)

    input_constraints = []
    if constraints is not None:
        input_constraints = model.map_constraints(constraints)

    start_hyp = model.start_hypothesis(mapped_inputs, input_constraints)

    beam_size = max(n_best, beam_size)
    # TODO -- working: switch to auto-scaling dynamic length factor logic for long constraints
    max_length = int(round(len(mapped_inputs[0][0]) * length_factor))
    logger.info('max_length: {}'.format(max_length))
    search_grid = decoder.search(start_hyp=start_hyp, constraints=input_constraints,
                                 max_hyp_len=max_length,
                                 beam_size=beam_size)

    best_output, best_alignments = decoder.best_n(search_grid, model.eos_token,
                                                  n_best=n_best,
                                                  return_model_scores=False,
                                                  return_alignments=True,
                                                  length_normalization=True,
                                                  prefer_eos=True)

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
    app.run(debug=False, port=port, host='127.0.0.1', threaded=False)

