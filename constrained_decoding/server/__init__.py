import logging
import os
import codecs
import re
from subprocess import Popen, PIPE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    def __init__(self, lang, use_subword=False, subword_codes=None, escape_special_chars=False, truecase_model=None):
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

        self.escape_special_chars = escape_special_chars

        if self.escape_special_chars:
            escape_special_chars_script = os.path.join(os.path.dirname(__file__),
                                                       'resources/tokenizer/escape-special-chars.perl')
            self.escape_special_chars_cmd = [escape_special_chars_script]

            deescape_special_chars_script = os.path.join(os.path.dirname(__file__),
                                                         'resources/tokenizer/deescape-special-chars.perl')
            self.deescape_special_chars_cmd = [deescape_special_chars_script]

        # WORKING: make quicker escape/descape implementation
        self.special_token_map = {
                             u'|': u'&#124;',
                             u'<': u'&lt;',
                             u'>': u'&gt;',
                             u'[': u'&bra;',
                             u']': u'&ket;',
                             u'"': u'&quot;',
                             u'\'': u'&apos;',
                             u'&': u'&amp;'}
        self.special_token_unmap = {v:k for k,v in self.special_token_map.items()}

        self.truecase = False
        if truecase_model is not None:
            self.truecase = True

            truecase_script = os.path.join(os.path.dirname(__file__),
                                           'resources/recaser/truecase.perl')
            self.truecase_cmd = [truecase_script, '-m', truecase_model]

            detruecase_script = os.path.join(os.path.dirname(__file__),
                                             'resources/recaser/detruecase.perl')
            self.detruecase_cmd = [detruecase_script]

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
        segment = segment.rstrip()

        if self.escape_special_chars:
            segment = segment.decode('utf8')
            for k, v in self.special_token_map.items():
                segment = re.sub(re.escape(k), v, segment)
            segment.encode('utf8')

            # char_escape = Popen(self.escape_special_chars_cmd, stdin=PIPE, stdout=PIPE)
            # this script cuts off a whitespace, so we add some extra
            # segment, _ = char_escape.communicate(segment + '   ')
            # segment = segment.rstrip()

        if self.truecase:
            # hack to make this faster
            segment = segment[0].lower() + segment[1:]
            # truecaser = Popen(self.truecase_cmd, stdin=PIPE, stdout=PIPE)
            # this script cuts off a whitespace, so we add some extra
            # segment, _ = truecaser.communicate(segment + '   ')
            # segment = segment.rstrip()

        if type(segment) is not unicode:
            utf_line = segment.decode('utf8')
        else:
            utf_line = segment

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

    def deescape_special_chars(self, text):
        for k, v in self.special_token_unmap.items():
            text = re.sub(re.escape(k), v, text)

        # if type(text) is unicode:
        #     text = text.encode('utf8')
        # char_deescape = Popen(self.deescape_special_chars_cmd, stdin=PIPE, stdout=PIPE)
        # this script cuts off a whitespace, so we add some extra
        # text, _ = char_deescape.communicate(text + '   ')
        # text = text.rstrip()
        # utf_line = text.decode('utf8')
        return text

    def detruecase(self, text):
        # hack to make this faster
        text = text[0].upper() + text[1:]
        return text

        # if type(text) is unicode:
        #     text = text.encode('utf8')

        # detruecaser = Popen(self.detruecase_cmd, stdin=PIPE, stdout=PIPE)
        # this script cuts off a whitespace, so we add some extra
        # text, _ = detruecaser.communicate(text + '   ')
        # text = text.rstrip()
        # utf_line = text.decode('utf8')
        # return utf_line

